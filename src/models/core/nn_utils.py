# nn_utils.py

import torch
from torch import nn
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

def cosine_sim(x1, x2=None, eps=1e-8):
    """
    Return pair-wise cosine similarity between each row in x1 and each row in x2.

    Args:
        x1 ([A, B])
        x2 ([C, B])

    Returns:
        [A, C]
    """
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

class AccessibleDataParallel(nn.DataParallel):
    """
    After wrapping a Module with DataParallel, the attributes of the module (e.g. custom methods) became inaccessible.
    This is because DataParallel defines a few new members, and allowing other attributes might
    lead to clashes in their names. See the following:
    https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

    This is a simple workaround class
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

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
