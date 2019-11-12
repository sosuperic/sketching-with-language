# nn_utils.py

import torch
import torch.nn.functional as F

#
# General
#

def move_to_cuda(x):
    """Move tensor to cuda"""
    if torch.cuda.is_available():
        if type(x) == tuple:
            x = tuple([t.cuda() for t in x])
        else:
            x = x.cuda()
    return x

def setup_seeds(seed=1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#
# Decoding utils
#
def logits_to_prob(logits, method='softmax',
                   tau=1.0, eps=1e-10, gumbel_hard=False):
    """
    Args:
        logits: [batch_size, vocab_size]
        method: 'gumbel', 'softmax'
        gumbel_hard: boolean
        topk: int (used for beam search)
    Returns: [batch_size, vocab_size]
    """
    if tau == 0.0:
        raise ValueError(
            'Temperature should not be 0. If you want greedy decoding, pass "greedy" to prob_to_vocab_id()')
    if method == 'gumbel':
        prob = F.gumbel_softmax(logits, tau=tau, eps=eps, hard=gumbel_hard)
    elif method == 'softmax':
        prob = F.softmax(logits / tau, dim=1)
    return prob

def prob_to_vocab_id(prob, method, k=10):
    """
    Produce vocab id given probability distribution over vocab
    Args:
        prob: [batch_size, vocab_size]
        method: str ('greedy', 'sample')
        k: int (used for beam search, sampling)
    Returns:
        prob: [batch_size * k, vocab_size]
            Rows are repeated:
                [[0.3, 0.2, 0.5],
                 [0.1, 0.7, 0.2]]
            Becomes (with k=2):
                [[0.3, 0.2, 0.5],
                 [0.3, 0.2, 0.5],
                 [0.1, 0.7, 0.2]
                 [0.1, 0.7, 0.2]]
        ids: [batch_size, k] LongTensor
    """
    if method == 'greedy':
        _, ids = torch.topk(prob, 1, dim=1)
    elif method == 'sample':
        ids = torch.multinomial(prob, k)
    return prob, ids


def lstm_generate(lstm, vocab_out_fc, tokens_embedding,
                  init_ids=None, hidden=None, cell=None,
                  condition_on_hc=True,
                  pad_id=None, eos_id=None,
                  max_len=100,
                  decode_method=None, tau=None, k=None,
                  idx2token=None,
                  ):
    """
    Decode up to max_len symbols by feeding previous output as next input.

    Args:
        lstm: nn.LSTM (hidden state is 
        vocab_out_fc: module used to transform Transformer's decoder outputs to logits
        tokens_embedding: nn.Embedding(vocab, dim)
        init_ids:   # [init_len, bsz]
        init_embs: [init_len, bsz, emb] (e.g. embedded SOS ids)
        hidden: [layers * direc, bsz, dim]
        cell: [layers * direc, bsz, dim]
        condition_on_hc: bool (condition on hidden and cell every time step)
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
    init_len, bsz = init_ids.size()
    vocab_size = len(idx2token)

    # Track which sequences have generated eos_id
    rows_with_eos = move_to_cuda(torch.zeros(bsz).long())
    pad_ids = move_to_cuda(torch.Tensor(bsz).fill_(pad_id)).long()
    pad_prob = move_to_cuda(torch.zeros(bsz, vocab_size))  # one hot for pad id
    pad_prob[:, pad_id] = 1

    # Generate
    decoded_probs = move_to_cuda(torch.zeros(bsz, max_len, vocab_size))  #
    decoded_ids = move_to_cuda(torch.zeros(bsz, max_len).long())  # [bsz, max_len]
    cur_input_id = init_ids
    for t in range(max_len):
        cur_input_emb = tokens_embedding(cur_input_id)  # [1, bsz, dim]
        if condition_on_hc:
            last_hc = hidden[-1,:,:] + cell[-1,:,:]  # [bsz, dim]
            last_hc = last_hc.unsqueeze(0)  # [1, bsz, dim]
            cur_input_emb = torch.cat([cur_input_emb, last_hc], dim=2)  # [1, bsz, dim * 2]
        dec_outputs, (hidden, cell) = lstm(cur_input_emb, (hidden, cell))  # [cur_len, bsz, dim]; h/c

        # Compute logits over vocab, use last output to get next token
        logits = vocab_out_fc(dec_outputs)  # [cur_len, bsz, vocab]
        logits.transpose_(0, 1)  # [bsz, cur_len, vocab]
        logits = logits[:, -1, :]  # last output; [bsz, vocab]
        prob = logits_to_prob(logits, tau=tau)  # [bsz, vocab]
        prob, ids = prob_to_vocab_id(prob, decode_method, k=k)  # prob: [bsz, vocab]; ids: [bsz, k]
        ids = ids[:, 0]  # get top k; [bsz]

        # If sequence (row) has already produced an eos_id, replace id with pad (and the prob with pad_prob)
        rows_with_eos = rows_with_eos | (ids == eos_id).long()
        prob = torch.where((rows_with_eos == 1).unsqueeze(1), pad_prob, prob)  # unsqueeze to broadcast
        ids = torch.where(rows_with_eos == 1, pad_ids, ids)

        # Update generated sequence so far
        decoded_probs[:,t,:] = prob
        decoded_ids[:,t] = ids

        cur_input_id = ids.unsqueeze(0)  # [1, bsz]

        # Terminate early if all sequences have generated eos
        if rows_with_eos.sum().item() == bsz:
            break

    # TODO: sort out init wonkiness
    # Remove initial input to decoder
    # decoded_probs = decoded_probs[:, init_embs.size(1):, :]
    # decoded_ids = decoded_ids[:, init_embs.size(1):]

    # Convert to strings
    decoded_texts = []
    if idx2token is not None:
        for i in range(bsz):
            tokens = []
            for j in range(decoded_ids.size(1)):
                id = decoded_ids[i][j].item()
                if id == eos_id:
                    break
                tokens.append(idx2token[id])
            text = ' '.join(tokens)
            decoded_texts.append(text)

    return decoded_probs, decoded_ids, decoded_texts