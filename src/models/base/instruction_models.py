# instruction_models.py

"""
Instruction (annotations from MTurk) related models and dataset 
"""

import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from src import utils
from src.data_manager.quickdraw import  LABELED_PROGRESSION_PAIRS_PATH, LABELED_PROGRESSION_PAIRS_DATA_PATH, \
    build_category_index, normalize_strokes, stroke3_to_stroke5
from src.models.core import nn_utils


LABELED_PROGRESSION_PAIRS_TRAIN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'train.pkl')
LABELED_PROGRESSION_PAIRS_VALID_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'valid.pkl')
LABELED_PROGRESSION_PAIRS_TEST_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'test.pkl')

LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2token.pkl')
LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'token2idx.pkl')
LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2cat.pkl')
LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'cat2idx.pkl')



##############################################################################
#
# DATASET
#
##############################################################################

PAD_ID, OOV_ID, SOS_ID, EOS_ID = 0, 1, 2, 3 # TODO: this should be a part of dataset maybe?

def build_vocab(data):
    """
    Returns mappings from index to token and vice versa.
    
    Args:
        data: list of dicts, each dict is one example.
    """
    tokens = set()
    for sample in data:
        text = utils.normalize_sentence(sample['annotation'])
        for token in text:
            tokens.add(token)

    idx2token = {}
    tokens = ['PAD', 'OOV', 'SOS', 'EOS'] + list(tokens)
    for i, token in enumerate(tokens):
        idx2token[i] = token
    token2idx = {v:k for k, v in idx2token.items()}

    return idx2token, token2idx


def save_progression_pair_dataset_splits_and_vocab():
    """
    Each split is a list of dicts, each dict is one example
    """
    tr_amt, val_amt, te_amt = 0.9, 0.05, 0.05

    # load data (saved by quickdraw.py)
    category_to_data = {}
    for fn in os.listdir(LABELED_PROGRESSION_PAIRS_DATA_PATH):
        category = os.path.splitext(fn)[0]  # cat.pkl
        fp = os.path.join(LABELED_PROGRESSION_PAIRS_DATA_PATH, fn)
        data = utils.load_file(fp)
        category_to_data[category] = data

    # split
    train, valid, test = [], [], []
    for category, data in category_to_data.items():
        l = len(data)
        tr_idx = int(tr_amt * l)
        val_idx = int((tr_amt + val_amt) * l)
        tr_data = data[:tr_idx]
        val_data = data[tr_idx:val_idx]
        te_data = data[val_idx:]
        train += tr_data
        valid += val_data
        test += te_data

    # save splits
    for data, fp in [(train, LABELED_PROGRESSION_PAIRS_TRAIN_PATH),
                     (valid, LABELED_PROGRESSION_PAIRS_VALID_PATH),
                     (test, LABELED_PROGRESSION_PAIRS_TEST_PATH)]:
        utils.save_file(data, fp)

    # build and save vocab
    idx2token, token2idx = build_vocab(train + valid + test)
    for data, fp in [(idx2token, LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH),
                     (token2idx, LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)]:
        utils.save_file(data, fp)

    # build and save category to index map (in case our model conditions on category)
    idx2cat, cat2idx = build_category_index(train + valid + test)
    for data, fp, in [(idx2cat, LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH),
                      (cat2idx, LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)]:
        utils.save_file(data, fp)


def map_sentence_to_index(sentence, token2idx):
    return [int(token2idx[tok]) for tok in utils.normalize_sentence(sentence)]


class ProgressionPairDataset(Dataset):
    def __init__(self, dataset_split, use_prestrokes=False):
        """
        
        Args:
            dataset_split: str
            remove_question_marks: bool (whether to remove samples where annotation was '?')
        """
        super().__init__()
        self.dataset_split = dataset_split
        self.use_prestrokes = use_prestrokes

        # Get data
        fp = None
        if dataset_split == 'train':
            fp = LABELED_PROGRESSION_PAIRS_TRAIN_PATH
        elif dataset_split == 'valid':
            fp = LABELED_PROGRESSION_PAIRS_VALID_PATH
        elif dataset_split == 'test':
            fp = LABELED_PROGRESSION_PAIRS_TEST_PATH
        if not os.path.exists(fp):  # create splits and vocab first time
            save_progression_pair_dataset_splits_and_vocab()
        data = utils.load_file(fp)

        # Load vocab and category mappings
        self.idx2token = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.vocab_size = len(self.idx2token)

        self.idx2cat = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH)
        self.cat2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)

        # TODO: should I be using stroke3_SEGMENT for the factor or stroke3? or
        # pass in the factor computed on the entire dataset?
        # TODO: Probably should just pass ins cale factor on entire sketch rnn data (which is already precomputed
        # and in stroke_models.py
        self.data = normalize_strokes(data, scale_factor_key='stroke3_segment',
                                      stroke_keys=['stroke3', 'stroke3_segment'])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get subsequence of drawing that was annotated
        stroke3 = sample['stroke3_segment']
        stroke5 = stroke3_to_stroke5(stroke3)

        if self.use_prestrokes:
            # Get subsequence that precedes the annotated
            stroke3_pre = sample['stroke3'][:sample['stroke3_start'],:]
            # Insert element so that all presegments have length at least 1
            stroke3_pre = np.vstack([np.array([0, 0, 1]), stroke3_pre])  # 1 for penup

            # Separator point is [0,0,1,1,0]. Should be identifiable as pt[2] is pen down, pt[3] is pen up and
            # doesn't occur in the data otherwise
            # TODO: is this a good separator token?
            sep_pt = np.array([0,0,1,1,0])
            stroke5_pre = stroke3_to_stroke5(stroke3_pre)
            stroke5 = np.vstack([stroke5_pre, sep_pt, stroke5])


        # Map
        text = sample['annotation']
        text_indices = map_sentence_to_index(text, self.token2idx)
        text_indices = [SOS_ID] + text_indices + [EOS_ID]

        # Additional metadata
        cat = sample['category']
        cat_idx = self.cat2idx[cat]
        url = sample['url']

        return (stroke5, text, text_indices, cat, cat_idx, url)

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch
        
        TODO: why did I write my own collate_fn? Is there something wrong with not using one for the StrokeDataset?
        
        Args:
            batch: list of samples, one sample is returned from __getitem__(idx)
        """
        strokes, texts, texts_indices, cats, cats_idx, urls = zip(*batch)
        bsz = len(batch)
        sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        # Create array of strokes, zeros for padding
        stroke_lens = [stroke.shape[0] for stroke in strokes]
        max_stroke_len = max(stroke_lens)
        batch_strokes = np.zeros((bsz, max_stroke_len, sample_dim))
        for i, stroke in enumerate(strokes):
            l = stroke.shape[0]
            batch_strokes[i,:l,:] = stroke

        # Create array of text indices, zeros for padding
        text_lens = [len(t) for t in texts_indices]
        max_text_len = max(text_lens)
        batch_text_indices = np.zeros((bsz, max_text_len))
        for i, text_indices in enumerate(texts_indices):
            l = len(text_indices)
            batch_text_indices[i,:l] = text_indices

        # Convert to Tensors
        batch_strokes = torch.FloatTensor(batch_strokes)
        batch_text_indices = torch.LongTensor(batch_text_indices)
        cats_idx = torch.LongTensor(cats_idx)

        return batch_strokes, stroke_lens, \
               texts, text_lens, batch_text_indices, cats, cats_idx, urls

##############################################################################
#
# MODEL
#
##############################################################################

class InstructionDecoderLSTM(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True,
                 condition_on_hc=False, use_categories=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim,
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.condition_on_hc = condition_on_hc
        self.use_categories = use_categories

        self.dropout_mod = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers, dropout=dropout, batch_first= batch_first)

    def forward(self, texts_emb, text_lens, hidden=None, cell=None,
                token_embedding=None, category_embedding=None, categories=None):
        """
        Args:
            texts_emb: [len, bsz, dim] FloatTensor
            text_lens: list of ints, length len
            hidden: [n_layers * n_directions, bsz, dim]  FloatTensor
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
            token_embedding: nn.Embedding(vocab, dim)
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor
            
        Returns:
            outputs:
                if token_embedding is None: [len, bsz, dim] FloatTensor 
                else: [len, bsz, vocab] FloatTensor 
            hidden: [n_layers * n_directions, bsz, dim]
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
        """

        # Condition on last layer's hidden and cell on every time step by combining last hidden and cell,
        # repeating along time dimension, and concatenating with encoded texts in feature dimension
        if self.condition_on_hc:
            last_hidden, last_cell = hidden[-1, :, :], cell[-1, :, :]  # last = [bsz, dim]
            last_hc = (last_hidden + last_cell).unsqueeze(0)  # [1, bsz, dim]
            last_hc = last_hc.repeat(texts_emb.size(0), 1, 1)  # [len, bsz, dim]
            inputs_emb = torch.cat([texts_emb, last_hc], dim=2)  # [len, bsz, dim * 2]
        else:
            inputs_emb = texts_emb

        if self.use_categories and category_embedding:
            cats_emb = category_embedding(categories)  # [bsz, dim]
            cats_emb = self.dropout_mod(cats_emb)
            cats_emb = cats_emb.repeat(inputs_emb.size(0), 1, 1)  # [len, bsz, dim]
            inputs_emb = torch.cat([inputs_emb, cats_emb], dim=2)  # [len, bsz, dim * 2 or dim *3]

        # decode
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs_emb, text_lens, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed_inputs, (hidden, cell))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # [max_text_len + 2, bsz, dim]; h/c = [n_layers * n_directions, bsz, dim]

        if token_embedding is not None:
            outputs = torch.matmul(outputs, token_embedding.weight.t())  # [len, bsz, vocab]

        return outputs, (hidden, cell)

    def generate(self,
                 token_embedding,
                 category_embedding=None, categories=None,
                 init_ids=None, hidden=None, cell=None,
                 pad_id=None, eos_id=None,
                 max_len=100,
                 decode_method=None, tau=None, k=None,
                 idx2token=None,
                 ):
        """
        Decode up to max_len symbols by feeding previous output as next input.

        Args:
            lstm: nn.LSTM
            token_embedding: nn.Embedding(vocab, dim)
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor
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
            cats_emb: [bsz, dim]

       Returns:
            decoded_probs: [bsz, max_len, vocab]
            decoded_ids: [bsz, max_len]
            decoded_texts: list of strs
        """
        init_len, bsz = init_ids.size()
        vocab_size = len(idx2token)

        # Track which sequences have generated eos_id
        rows_with_eos = nn_utils.move_to_cuda(torch.zeros(bsz).long())
        pad_ids = nn_utils.move_to_cuda(torch.Tensor(bsz).fill_(pad_id)).long()
        pad_prob = nn_utils.move_to_cuda(torch.zeros(bsz, vocab_size))  # one hot for pad id
        pad_prob[:, pad_id] = 1

        # Generate
        decoded_probs = nn_utils.move_to_cuda(torch.zeros(bsz, max_len, vocab_size))  #
        decoded_ids = nn_utils.move_to_cuda(torch.zeros(bsz, max_len).long())  # [bsz, max_len]
        cur_input_id = init_ids
        cats_emb = category_embedding(categories).unsqueeze(0) if (category_embedding is not None) else None  # [1, bsz, dim]
        for t in range(max_len):
            cur_input_emb = token_embedding(cur_input_id)  # [1, bsz, dim]
            if self.condition_on_hc:
                last_hc = hidden[-1, :, :] + cell[-1, :, :]  # [bsz, dim]
                last_hc = last_hc.unsqueeze(0)  # [1, bsz, dim]
                cur_input_emb = torch.cat([cur_input_emb, last_hc], dim=2)  # [1, bsz, dim * 2]
            if (cats_emb is not None):
                cur_input_emb = torch.cat([cur_input_emb, cats_emb], dim=2)  # 1, bsz, dim * 2 or dim * 3]

            dec_outputs, (hidden, cell) = self.lstm(cur_input_emb, (hidden, cell))  # [cur_len, bsz, dim]; h/c

            # Compute logits over vocab, use last output to get next token
            # TODO: can we use self.forward
            logits = torch.matmul(dec_outputs, token_embedding.weight.t())  # [cur_len, bsz, vocab]
            logits.transpose_(0, 1)  # [bsz, cur_len, vocab]
            logits = logits[:, -1, :]  # last output; [bsz, vocab]
            prob = nn_utils.logits_to_prob(logits, tau=tau)  # [bsz, vocab]
            prob, ids = nn_utils.prob_to_vocab_id(prob, decode_method, k=k)  # prob: [bsz, vocab]; ids: [bsz, k]
            ids = ids[:, 0]  # get top k; [bsz]

            # Update generated sequence so far
            # If sequence (row) has already produced an eos_id *earlier*, replace id/prob with pad
            # TODO: I don't think decoded_probs is being filled with pad_prob for some reason
            prob = torch.where((rows_with_eos == 1).unsqueeze(1), pad_prob, prob)  # unsqueeze to broadcast
            ids = torch.where(rows_with_eos == 1, pad_ids, ids)
            decoded_probs[:, t, :] = prob
            decoded_ids[:, t] = ids

            # Update for next iteration in loop
            rows_with_eos = rows_with_eos | (ids == eos_id).long()
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