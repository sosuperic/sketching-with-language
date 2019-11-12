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

import src.utils as utils


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=250):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x

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

        src_key_padding_mask = utils.move_to_cuda(src_key_padding_mask)
        memory_key_padding_mask = utils.move_to_cuda(memory_key_padding_mask)

    # Tgt mask
    if tgt_lens is not None:
        max_tgt_len = max(tgt_lens)
        tgt_key_padding_mask = torch.zeros(bsz, max_tgt_len).bool()
        for i, seq_len in enumerate(tgt_lens):
            tgt_key_padding_mask[i, seq_len:] = 1

        tgt_key_padding_mask = utils.move_to_cuda(tgt_key_padding_mask)

    return src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

def generate_square_subsequent_mask(size):
    """
    Generate a square mask for the sequence that prevents attending to items in the future.

    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = utils.move_to_cuda(mask)

    return mask

def scale_add_pos_emb(input_emb, pos_enc):
    """
    Scale input embs and add position encoding.
    
    :param input_emb: [bsz, seq_len, dim]
    :param pos_enc: [bsz, pos_len, dim] (pos_len must be greater than seq_len)
    :return: [bsz, seq_len, dim]
    """
    input_emb *= math.sqrt(input_emb.size(-1))
    input_emb += pos_enc[:, :input_emb.size(1), :]
    return input_emb

def generate(transformer, vocab_out_fc, tokens_embedding, pos_enc,
             input_embs=None, input_lens=None,
             init_ids=None, init_embs=None,
             PAD_ID=None, EOS_ID=None,
             max_len=100,
             decode_method='sample', tau=1.0, k=1,
             idx2token=None,
             ):
    """
    Decode up to max_len symbols by feeding previous output as next input.
    
    Args:
        transformer: nn.Transformer
        vocab_out_fc: module used to transform Transformer's decoder outputs to logits
        tokens_embedding: nn.Embedding(vocab, dim)
        input_embs: [bsz, input_len, dim]
        input_lens: list of ints
        init_ids: [bsz, init_len]  (e.g. SOS ids)
        init_embs: [bsz, init_len, emb] (e.g. embedded SOS ids)
        EOS_ID: int (id for EOS_ID token)
        decode_method: str (how to sample words given probabilities; 'greedy', 'sample')
        tau: float (temperature for softmax)
        k: int (for sampling or beam search)
        idx2token: dict

   Returns:
        decoded_probs: [bsz, max_len, vocab]
        decoded_ids: [bsz, max_len]
        decoded_texts: list of strs
    """
    bsz = init_embs.size(0)
    vocab_size = len(idx2token)

    # Encode inputs
    src_key_padding_mask, _ , memory_key_padding_mask = create_transformer_padding_masks(src_lens=input_lens)
    input_embs.transpose_(0,1)  # [input_len, bsz, dim]
    memory = transformer.encoder(input_embs, src_key_padding_mask=src_key_padding_mask)  # [input_len, bsz, dim]

    # Track which sequences have generated eos_id
    rows_with_eos = utils.move_to_cuda(torch.zeros(bsz).long())
    pad_ids = utils.move_to_cuda(torch.Tensor(bsz).fill_(PAD_ID)).long()
    pad_prob = utils.move_to_cuda(torch.zeros(bsz, vocab_size))  # one hot for pad id
    pad_prob[:, PAD_ID] = 1

    # Generate
    decoded_probs = utils.move_to_cuda(torch.zeros(bsz, max_len, vocab_size))
    decoded_ids = init_ids
    for i in range(max_len):
        # pass through TransformerDecoder
        tgt_mask = generate_square_subsequent_mask(decoded_ids.size(1)).type_as(decoded_ids)
        decoded_ids_emb = tokens_embedding(decoded_ids)  # [bsz, cur_len, dim]
        decoded_ids_emb.transpose_(0,1)  # [cur_len, bsz, dim]
        decoded_ids_emb = scale_add_pos_emb(decoded_ids_emb, pos_enc)

        dec_outputs = transformer.decoder(decoded_ids_emb, memory,  # dec_outputs = [cur_len, bsz, dim]
                                          tgt_mask=tgt_mask, # TODO: why does this affect it?
                                          memory_key_padding_mask=memory_key_padding_mask)

        # Compute logits over vocab, use last output to get next token
        logits = vocab_out_fc(dec_outputs)  # [cur_len, bsz, vocab]
        logits.transpose_(0,1)      # [bsz, cur_len, vocab]
        logits = logits[:,-1,:]  # last output; [bsz, vocab]
        prob = utils.logits_to_prob(logits, tau=tau)  # [bsz, vocab]
        prob, ids = utils.prob_to_vocab_id(prob, decode_method, k=k)  # prob: [bsz, vocab]; ids: [bsz, k]
        ids = ids[:,0]  # get top k

        # If sequence (row) has already produced an EOS_ID, replace id with pad (and the prob with pad_prob)
        rows_with_eos = rows_with_eos | (ids == EOS_ID).long()
        prob = torch.where((rows_with_eos == 1).unsqueeze(1), pad_prob, prob)  # unsqueeze to broadcast
        ids = torch.where(rows_with_eos == 1, pad_ids, ids)

        # Update generated sequence so far
        decoded_probs = torch.cat([decoded_probs, prob.unsqueeze(1)], dim=1)  # [bsz, init_len + (t+1), vocab]
        decoded_ids = torch.cat([decoded_ids, ids.unsqueeze(1)], dim=1)  # [bsz, init_len + (t+1)]

        # Terminate early if all sequences have generated eos
        if rows_with_eos.sum().item() == bsz:
            break

    # Remove initial input to decoder
    decoded_probs = decoded_probs[:, init_embs.size(1):, :]
    decoded_ids = decoded_ids[:, init_embs.size(1):]

    # Convert to strings
    decoded_texts = []
    if idx2token is not None:
        for i in range(bsz):
            tokens = []
            for j in range(decoded_ids.size(1)):
                id = decoded_ids[i][j].item()
                if id == EOS_ID:
                    break
                tokens.append(idx2token[id])
            text = ' '.join(tokens)
            decoded_texts.append(text)

    return decoded_probs, decoded_ids, decoded_texts
