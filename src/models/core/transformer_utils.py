# transformer_utils.py

"""
The nn.transformer doesn't contain code for everything.
    - https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html

Additional utils found here
- positional encoding
- masking for decoding
- generation
"""

import math

import torch
import torch.nn as nn

import src.utils as utils
import src.models.core.nn_utils as nn_utils


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len=500, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)  # [len, 1, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [len, bsz, dim]
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)

def scale_add_pos_emb(input_emb, pos_enc):
    """
    Scale input embs and add position encoding.

    :param input_emb: [bsz, seq_len, dim]
    :param pos_enc: [bsz, pos_len, dim] (pos_len must be greater than seq_len)
    :return: [bsz, seq_len, dim]
    """
    input_emb *= math.sqrt(input_emb.size(-1))
    input_emb += pos_enc(input_emb)
    return input_emb

def create_transformer_padding_masks(src_lens=None, tgt_lens=None):
    """
    Return ByteTensors where a true value means value should be ignored. Used to handle variable length
    sequences within a batch.

    Args:
        src_lens: list of length bsz
        tgt_lens: list of length bsz

    Returns:
        src_key_padding_mask: [bsz, max_src_len] ByteTensor
        tgt_key_padding_mask: [bsz, max_tgt_len] ByteTensor
        memory_key_padding_mask: [bsz, max_src_len] ByteTensor
    """
    src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = None, None, None

    # Src and memory masks
    if src_lens is not None:
        bsz = len(src_lens)
        max_src_len = max(src_lens)
        src_key_padding_mask = torch.zeros(bsz, max_src_len).bool()
        for i, seq_len in enumerate(src_lens):
            src_key_padding_mask[i, seq_len:] = 1

        memory_key_padding_mask = src_key_padding_mask

        src_key_padding_mask = nn_utils.move_to_cuda(src_key_padding_mask)
        memory_key_padding_mask = nn_utils.move_to_cuda(memory_key_padding_mask)

    # Tgt mask
    if tgt_lens is not None:
        bsz = len(tgt_lens)
        max_tgt_len = max(tgt_lens)
        tgt_key_padding_mask = torch.zeros(bsz, max_tgt_len).bool()
        for i, seq_len in enumerate(tgt_lens):
            tgt_key_padding_mask[i, seq_len:] = 1

        tgt_key_padding_mask = nn_utils.move_to_cuda(tgt_key_padding_mask)

    return src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

def generate_square_subsequent_mask(size):
    """
    Generate a square mask for the sequence that prevents attending to items in the future.

    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)  # True's in lower left half
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = nn_utils.move_to_cuda(mask)

    return mask

def transformer_generate(
        transformer, token_embedding, pos_enc,
        src_input_embs=None, input_lens=None,
        init_ids=None,
        pad_id=None, eos_id=None,
        max_len=100,
        decode_method=None, tau=None, k=None,
        idx2token=None,
        ):
    """
    Decode up to max_len symbols by feeding previous output as next input.

    Args:
        transformer: nn.Transformer
        token_embedding: nn.Embedding(vocab, dim)
        pos_enc: PositionalEncoder module
        input_embs: [input_len, bsz, dim]
        input_lens: list of ints
        init_ids: [init_len, bsz]  (e.g. SOS ids)
        init_embs: [init_len, bsz, emb] (e.g. embedded SOS ids)
        pad_id: int
        eos_id: int (id for EOS_ID token)
        decode_method: str (how to sample words given probabilities; 'greedy', 'sample')
        tau: float (temperature for softmax)
        k: int (for sampling or beam search)
        idx2token: dict

   Returns:
        decoded_probs: [bsz, max_len, vocab]
        decoded_ids: [bsz, max_len]
        decoded_texts: list of strs
    """
    init_len, bsz = init_ids.size()
    vocab_size = len(idx2token)

    # Encode inputs
    src_key_padding_mask, _ , memory_key_padding_mask = create_transformer_padding_masks(src_lens=input_lens)
    memory = transformer.encoder(src_input_embs, src_key_padding_mask=src_key_padding_mask)  # [input_len, bsz, dim]

    # Track which sequences have generated eos_id
    rows_with_eos = nn_utils.move_to_cuda(torch.zeros(bsz).long())
    pad_ids = nn_utils.move_to_cuda(torch.Tensor(bsz).fill_(pad_id)).long()
    pad_prob = nn_utils.move_to_cuda(torch.zeros(bsz, vocab_size))  # one hot for pad id
    pad_prob[:, pad_id] = 1

    # Generate
    decoded_probs = nn_utils.move_to_cuda(torch.zeros(init_len + max_len, bsz, vocab_size))
    decoded_ids = nn_utils.move_to_cuda(torch.zeros(init_len + max_len, bsz).long())
    decoded_ids[:init_len, :] = init_ids
    for t in range(init_len, max_len):
        # pass through TransformerDecoder
        tgt_mask = generate_square_subsequent_mask(t).type_as(decoded_ids)
        cur_dec_input = decoded_ids[:t, :]  # [t, bsz]
        cur_dec_input = token_embedding(cur_dec_input)
        cur_dec_input = scale_add_pos_emb(cur_dec_input, pos_enc)  # [t, bsz, dim]

        dec_outputs = transformer.decoder(cur_dec_input, memory,  # dec_outputs = [t, bsz, dim]
                                          tgt_mask=tgt_mask,
                                          memory_key_padding_mask=memory_key_padding_mask
                                          )

        # Compute logits over vocab, use last output to get next token
        logits = torch.matmul(dec_outputs, token_embedding.weight.t())  # [t, bsz, vocab]
        logits = logits[-1,:,:]  # [bsz, vocab]
        prob = nn_utils.logits_to_prob(logits, tau=tau)  # [bsz, vocab]
        prob, ids = nn_utils.prob_to_vocab_id(prob, decode_method, k=k)  # prob: [bsz, vocab]; ids: [bsz, k]
        ids = ids[:,0]  # get top k

        # Update generated sequence so far
        # If sequence (row) has already produced an eos_id *earlier*, replace id/prob with pad
        # TODO: I don't think decoded_probs is being filled with pad_prob for some reason
        prob = torch.where((rows_with_eos == 1).unsqueeze(1), pad_prob, prob)  # unsqueeze to broadcast
        ids = torch.where(rows_with_eos == 1, pad_ids, ids)
        decoded_probs[t, :, :] = prob
        decoded_ids[t, :] = ids

        # Update for next iteration in loop
        rows_with_eos = rows_with_eos | (ids == eos_id).long()

        # Terminate early if all sequences have generated eos
        if rows_with_eos.sum().item() == bsz:
            break

    # # Remove initial input to decoder
    decoded_probs = decoded_probs[init_len:,:,:]
    decoded_ids = decoded_ids[init_len:,:]

    # TODO: remove this once InstructionDecoderLSTM is refactored to return [len, bsz] instead of [bsz, len]
    decoded_probs.transpose_(0,1)
    decoded_ids.transpose_(0,1)

    # Convert to strings
    decoded_texts = []
    if idx2token is not None:
        for i in range(bsz):
            tokens = []
            for j in range(decoded_ids.size(1)):
                id = decoded_ids[i][j].item()
                # import pdb; pdb.set_trace()  # TODO: Saw an example that was EOS EOS EOS... why isn't this being caught by the following equality statement?
                if id == eos_id:
                    break
                tokens.append(idx2token[id])
            text = ' '.join(tokens)
            decoded_texts.append(text)

    import pdb; pdb.set_trace()

    return decoded_probs, decoded_ids, decoded_texts
