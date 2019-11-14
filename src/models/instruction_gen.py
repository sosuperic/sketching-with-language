# instruction_gen.py

import argparse
import matplotlib
matplotlib.use('Agg')
from nltk.tokenize import word_tokenize  # TODO: add the download punkt to requirements.txt
import numpy as np
import os

from rouge_score import rouge_scorer

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.data_manager.quickdraw import LABELED_PROGRESSION_PAIRS_PATH, LABELED_PROGRESSION_PAIRS_DATA_PATH
from src.models.sketch_rnn import stroke3_to_stroke5, TrainNN
from src.models.train_nn import TrainNN, RUNS_PATH
from src.models.transformer_utils import *
import src.utils as utils
import src.models.nn_utils as nn_utils

LABELED_PROGRESSION_PAIRS_TRAIN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'train.pkl')
LABELED_PROGRESSION_PAIRS_VALID_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'valid.pkl')
LABELED_PROGRESSION_PAIRS_TEST_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'test.pkl')

LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2token.pkl')
LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'token2idx.pkl')
LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2cat.pkl')
LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'cat2idx.pkl')


USE_CUDA = torch.cuda.is_available()


##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        # Training
        self.batch_size = 64  # 100
        self.lr = 0.0001  # 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 1000

        # Model
        self.dim = 512
        self.n_enc_layers = 4
        self.n_dec_layers = 4
        self.model_type = 'cnn_lstm'  # 'lstm', 'transformer'
        self.condition_on_hc = False  # With 'lstm', input to decoder also contains last hidden cell
        self.use_prestrokes = True
        self.use_categories = True
        self.dropout = 0.2

        # inference
        self.decode_method = 'greedy'  # 'sample', 'greedy'
        self.tau = 1.0  # sampling text
        self.k = 5      # sampling text

        # Other
        self.notes = ''



##############################################################################
#
# DATASET
#
##############################################################################

PAD_ID, OOV_ID, SOS_ID, EOS_ID = 0, 1, 2, 3 # TODO: this should be a part of dataset maybe?

def normalize(sentence):
    """Tokenize"""
    return word_tokenize(sentence.lower())

def build_vocab(data):
    """
    Returns mappings from index to token and vice versa.
    
    :param data: list of dicts, each dict is one example.
    """
    tokens = set()
    for sample in data:
        text = normalize(sample['annotation'])
        for token in text:
            tokens.add(token)

    idx2token = {}
    tokens = ['PAD', 'OOV', 'SOS', 'EOS'] + list(tokens)
    for i, token in enumerate(tokens):
        idx2token[i] = token
    token2idx = {v:k for k, v in idx2token.items()}

    return idx2token, token2idx

def build_category_index(data):
    """
    Returns mappings from index to category and vice versa.
    
    :param data: list of dicts, each dict is one example
    """
    categories = set()
    for sample in data:
        categories.add(sample['category'])
    idx2cat = {i: cat for i, cat in enumerate(categories)}
    cat2idx = {cat: i  for i, cat in idx2cat.items()}

    return idx2cat, cat2idx

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

def map_str_to_index(s, token2idx):
    return [int(token2idx[tok]) for tok in normalize(s)]


def normalize_data(data):
    """
    Normalize entire dataset (delta_x, delta_y) by the scaling factor.

    :param data: list of dicts
    """
    scale_factor = calculate_normalizing_scale_factor(data)
    normalized_data = []
    for sample in data:
        stroke3_seg = sample['stroke3_segment']
        stroke3 = sample['stroke3']
        stroke3_seg[:, 0:2] /= scale_factor
        stroke3[:, 0:2] /= scale_factor
        sample['stroke3_segment'] = stroke3_seg
        sample['stroke3'] = stroke3
        normalized_data.append(sample)
    return normalized_data

def calculate_normalizing_scale_factor(data):  # calculate_normalizing_scale_factor() in sketch_rnn/utils.py
    """
    Calculate the normalizing factor in Appendix of paper

    :param data: list of dicts
    """
    deltas = []
    for sample in data:
        stroke = sample['stroke3_segment']
        for j in range(stroke.shape[0]):
            deltas.append(stroke[j][0])
            deltas.append(stroke[j][1])
    deltas = np.array(deltas)
    scale_factor = np.std(deltas)
    return scale_factor


class ProgressionPairDataset(Dataset):
    def __init__(self, dataset_split, remove_question_marks=False):
        """
        
        Args:
            dataset_split: str
            remove_question_marks: bool (whether to remove samples where annotation was '?')
        """
        super().__init__()
        self.dataset_split = dataset_split
        self.remove_quesiton_marks = remove_question_marks

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

        if remove_question_marks:
            new_data = []
            for sample in data:
                if sample['annotation'] != '?':
                    new_data.append(sample)
            data = new_data

        # Load vocab and category mappings
        self.idx2token = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.vocab_size = len(self.idx2token)

        self.idx2cat = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH)
        self.cat2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)

        self.data = normalize_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get subsequence of drawing that was annotated
        stroke3_seg = sample['stroke3_segment']
        stroke5_seg = stroke3_to_stroke5(stroke3_seg, len(stroke3_seg))
        # Get subsequence that precedes the annotated
        stroke3_pre_seg = sample['stroke3'][:sample['stroke3_start'],:]

        # Insert element so that all presegments have length at least 1
        stroke3_pre_seg = np.vstack([np.array([0, 0, 1]), stroke3_pre_seg])  # 1 for penup

        stroke5_pre_seg = stroke3_to_stroke5(stroke3_pre_seg, len(stroke3_pre_seg))

        # Map
        text = sample['annotation']
        text_indices = map_str_to_index(text, self.token2idx)
        text_indices = [SOS_ID] + text_indices + [EOS_ID]

        # Additional metadata
        cat = sample['category']
        cat_idx = self.cat2idx[cat]
        url = sample['url']

        return (stroke5_seg, stroke5_pre_seg, text, text_indices, cat, cat_idx, url)

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch
        
        :param: batch: list of samples, one sample is returned from __getitem__(idx)
        """
        strokes, prestrokes, texts, texts_indices, cats, cats_idx, urls = zip(*batch)
        bsz = len(batch)
        sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        # Create array of strokes, zeros for padding
        stroke_lens = [stroke.shape[0] for stroke in strokes]
        max_stroke_len = max(stroke_lens)
        batch_strokes = np.zeros((bsz, max_stroke_len, sample_dim))
        for i, stroke in enumerate(strokes):
            l = stroke.shape[0]
            batch_strokes[i,:l,:] = stroke

        # Create array of strokes, zeros for padding
        prestroke_lens = [prestroke.shape[0] for prestroke in prestrokes]
        max_prestrokes_len = max(prestroke_lens)
        batch_prestrokes = np.zeros((bsz, max_prestrokes_len, sample_dim))
        for i, prestrokes in enumerate(prestrokes):
            l = prestrokes.shape[0]
            batch_prestrokes[i, :l, :] = prestrokes

        # Create array of text indices, zeros for padding
        text_lens = [len(t) for t in texts_indices]
        max_text_len = max(text_lens)
        batch_text_indices = np.zeros((bsz, max_text_len))
        for i, text_indices in enumerate(texts_indices):
            l = len(text_indices)
            batch_text_indices[i,:l] = text_indices

        # Convert to Tensors
        batch_strokes = torch.FloatTensor(batch_strokes)
        batch_prestrokes = torch.FloatTensor(batch_prestrokes)
        batch_text_indices = torch.LongTensor(batch_text_indices)
        cats_idx = torch.LongTensor(cats_idx)

        return batch_strokes, stroke_lens, batch_prestrokes, prestroke_lens,\
               texts, text_lens, batch_text_indices, cats, cats_idx, urls



##############################################################################
#
# MODEL
#
##############################################################################

class StrokeEncoderCNN(nn.Module):
    def __init__(self, filter_sizes=[3,4,5], n_feat_maps=128, input_dim=None, emb_dim=None, dropout=None):
        """
        Args:
            filter_sizes: list of ints
                - Size of convolution window (referred to as filter widths in original paper)
            n_feat_maps: int
                - Number of output feature maps for each filter size
            input_dim: int (size of inputs)
            emb_dim: int (size of embedded inputs)
            dropout_prob: float
        """
        super().__init__()
        self.filter_sizes = filter_sizes
        self.n_feat_maps = n_feat_maps
        self.input_dim = input_dim

        self.input_fc = nn.Linear(input_dim, emb_dim)
        self.cnn_modlist = nn.ModuleList(
            [nn.Conv2d(1, n_feat_maps, (size, emb_dim)) for size in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, strokes, stroke_lens, prestrokes=None, prestroke_lens=None,
                category_embedding=None, categories=None):
        """
        Args:
            strokes: [seq_len, bsz, input_dim]

        Returns:  [bsz, dim]
        """
        strokes = self.input_fc(strokes)  # [seq_len, bsz, emb_dim]
        strokes = strokes.transpose(0,1).unsqueeze(1)  # [bsz, 1, seq_len, emb_dim]

        cnn_relus = [F.relu(cnn(strokes)) for cnn in self.cnn_modlist]
        # Each is [bsz, n_feat_maps, seq_len-filter_size+1, 1]

        # Pool over time dimension
        pooled = [F.max_pool2d(cnn_relu, (cnn_relu.size(2), 1)).squeeze(3).squeeze(2) for cnn_relu in cnn_relus]
        # Each is [batch, n_feat_maps]

        embedded = torch.stack([p.unsqueeze(0) for p in pooled], dim=0) # [n_filters, bsz, dim]
        embedded = embedded.mean(dim=0).squeeze(0)  # [bsz, dim]
        embedded = self.dropout(embedded)

        return embedded

class StrokeEncoderTransformer(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0,
                 use_prestrokes=False, use_categories=False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_prestrokes = use_prestrokes
        self.use_categories = use_categories

        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, 2, dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu'
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        if use_categories:
            self.dropout_mod = nn.Dropout(dropout)
            self.stroke_cat_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,  strokes, stroke_lens,
                prestrokes=None, prestroke_lens=None,
                category_embedding=None, categories=None):
        """
        Args:
            strokes:  [max_stroke_len, bsz, input_dim]
            stroke_lens: list of ints, length max_stroke_len
            prestrokes:  [max_prestroke_len, bsz]
            prestroke_lens: list of ints, length max_prestroke_len
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor

        Returns:
            hidden: [bsz, dim]
        """
        # TODO: using prestrokes is a bit annoying...
        # We can't simply 1) concat strokes and prestrokes, 2) add stroke_lens and prestroke_lens
        # because there would be padding elements between strokes and prestrokes.
        bsz = strokes.size(1)

        # if category_embedding:
        #     cats_emb =  category_embedding(categories)  # [bsz, dim]
        #     cats_emb = self.dropout_mod(cats_emb)
        #     strokes = torch.cat([strokes, cats_emb.repeat(strokes.size(0), 1, 1)], dim=2)  # [len, bsz, input+hidden]
        #     strokes = self.stroke_cat_fc(strokes)  # [len, bsz, hidden]
        #     if self.use_prestrokes:
        #         prestrokes = torch.cat([prestrokes, cats_emb.repeat(prestrokes.size(0), 1, 1)], dim=2)
        #         prestrokes = self.stroke_cat_fc(prestrokes)

        strokes = self.input_fc(strokes)  # [len, bsz, hsz]

        strokes_pad_mask, _, _ = create_transformer_padding_masks(src_lens=stroke_lens)
        memory = self.enc(strokes, src_key_padding_mask=strokes_pad_mask)  # [len, bsz, dim]

        hidden = []
        for i in range(bsz):  # TODO: what is a tensor op to do this?
            item_len = stroke_lens[i]
            item_emb = memory[:item_len,i,:].mean(dim=0)  # [dim]
            hidden.append(item_emb)
        hidden = torch.stack(hidden, dim=0)  # [bsz, dim]

        return hidden

class StrokeEncoderLSTM(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True,
                 use_prestrokes=False, use_categories=False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_prestrokes = use_prestrokes
        self.use_categories = use_categories

        if use_categories:
            self.dropout_mod = nn.Dropout(dropout)
            self.stroke_cat_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim,
                                bidirectional=True,
                                num_layers=num_layers, dropout=dropout, batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim,
                                num_layers=num_layers, dropout=dropout, batch_first=batch_first)

    def forward(self, strokes, stroke_lens,
                prestrokes=None, prestroke_lens=None,
                category_embedding=None, categories=None):
        """
        Args:
            strokes:  [max_stroke_len, bsz, input_dim]
            stroke_lens: list of ints, length max_stroke_len
            prestrokes:  [max_prestroke_len, bsz]
            prestroke_lens: list of ints, length max_prestroke_len
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor

        Returns:
            stroke_outputs: [max_stroke_len, bsz, dim]
            prestroke_outputs: [max_prestroke_len, bsz, dim]
            hidden: [layers * direc, bsz, dim]
            cell:  [layers * direc, bsz, dim]
        """
        # Compute a category embedding, repeat it along the time dimension, concatenate it with the strokes along
        # the feature dimension, and apply a fully connected
        bsz = strokes.size(1)

        if category_embedding:
            cats_emb =  category_embedding(categories)  # [bsz, dim]
            cats_emb = self.dropout_mod(cats_emb)
            strokes = torch.cat([strokes, cats_emb.repeat(strokes.size(0), 1, 1)], dim=2)  # [len, bsz, input+hidden]
            strokes = self.stroke_cat_fc(strokes)  # [len, bsz, hidden]
            if self.use_prestrokes:
                prestrokes = torch.cat([prestrokes, cats_emb.repeat(prestrokes.size(0), 1, 1)], dim=2)
                prestrokes = self.stroke_cat_fc(prestrokes)

        if self.use_prestrokes:
            packed_prestrokes = nn.utils.rnn.pack_padded_sequence(prestrokes, prestroke_lens, enforce_sorted=False)
            _, (pre_h, pre_c) = self.lstm(packed_prestrokes)  # [max_pre_len, bsz, dim]; h/c = [layers * direc, bsz, dim]

            packed_strokes = nn.utils.rnn.pack_padded_sequence(strokes, stroke_lens, enforce_sorted=False)
            strokes_outputs, (hidden, cell) = self.lstm(packed_strokes, (pre_h, pre_c))  # [max_stroke_len, bsz, dim]; h/c = [layers * direc, bsz, dim]
            strokes_outputs, _ = nn.utils.rnn.pad_packed_sequence(strokes_outputs)
        else:
            packed_strokes = nn.utils.rnn.pack_padded_sequence(strokes, stroke_lens, enforce_sorted=False)
            strokes_outputs, (hidden, cell) = self.lstm(packed_strokes)
            strokes_outputs, _ = nn.utils.rnn.pad_packed_sequence(strokes_outputs)

        # Take mean along num_directions because decoder is unidirectional lstm (this is bidirectional)
        hidden = hidden.view(self.num_layers, 2, bsz, self.hidden_dim).mean(dim=1)  # [layers, bsz, dim]
        cell = cell.view(self.num_layers, 2, bsz, self.hidden_dim).mean(dim=1)  # [layers, bsz, dim]

        return strokes_outputs, (hidden, cell)

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

class StrokeToInstructionModel(TrainNN):
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        self.tr_loader, self.val_loader = self.get_data_loaders()

        # Model
        self.token_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.dim)
        self.models.append(self.token_embedding)
        self.category_embedding = None
        if self.hp.use_categories:
            self.category_embedding = nn.Embedding(35, self.hp.dim)
            self.models.append(self.category_embedding)

        if hp.model_type.endswith('lstm'):
            # encoders may be different
            if hp.model_type == 'cnn_lstm':
                if hp.use_prestrokes:
                    raise NotImplementedError('Using prestrokes not implemented for cnn_lstm')
                self.enc = StrokeEncoderCNN(n_feat_maps=hp.dim, input_dim=5, emb_dim=hp.dim, dropout=hp.dropout)
            elif hp.model_type == 'transformer_lstm':
                self.enc = StrokeEncoderTransformer(
                    5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout,
                    use_prestrokes=hp.use_prestrokes, use_categories=hp.use_categories,
                )
                if hp.use_prestrokes:
                    raise NotImplementedError('Using prestrokes not implemented for transformer_lstm')
            elif hp.model_type == 'lstm':
                self.enc = StrokeEncoderLSTM(
                    5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout, batch_first=False,
                    use_prestrokes=hp.use_prestrokes, use_categories=hp.use_categories,
                )

            # decoder is lstm
            dec_input_dim = hp.dim
            if hp.condition_on_hc:
                dec_input_dim += hp.dim
            if hp.use_categories:
                dec_input_dim += hp.dim
            self.dec = InstructionDecoderLSTM(
                dec_input_dim, hp.dim, num_layers=hp.n_dec_layers, dropout=hp.dropout, batch_first=False,
                condition_on_hc=hp.condition_on_hc, use_categories=hp.use_categories
            )

            self.models.extend([self.enc, self.dec])
        elif hp.model_type == 'transformer':
            if hp.use_prestrokes:
                raise NotImplementedError('Using prestrokes not implemented for Transformer')
            if hp.use_categories:
                raise NotImplementedError('Use categories not implemented for Transformer')

            self.strokes_input_fc = nn.Linear(5, hp.dim)
            self.pos_enc = PositionalEncoder(hp.dim, max_seq_len=250)
            self.transformer = nn.Transformer(
                d_model=hp.dim, dim_feedforward=hp.dim * 4, nhead=2, activation='relu',
                num_encoder_layers=hp.n_enc_layers, num_decoder_layers=hp.n_dec_layers,
                dropout=hp.dropout,
            )
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.models.extend([self.strokes_input_fc, self.pos_enc, self.transformer])


        for model in self.models:
            model.cuda()

        # Optimizers
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    #
    # Data
    #
    def get_data_loaders(self):
        tr_dataset = ProgressionPairDataset('train')
        val_dataset = ProgressionPairDataset('valid')
        tr_loader = DataLoader(tr_dataset, batch_size=self.hp.batch_size, shuffle=True,
                               collate_fn=ProgressionPairDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.hp.batch_size, shuffle=False,
                                collate_fn=ProgressionPairDataset.collate_fn)
        return tr_loader, val_loader

    def preprocess_batch_from_data_loader(self, batch):
        """
        Convert tensors to cuda and convert to [len, bsz, ...] instead of [bsz, len, ...]
        """
        preprocessed = []
        for item in batch:
            if type(item) == torch.Tensor:
                item = nn_utils.move_to_cuda(item)
                if item.dim() > 1:
                    item.transpose_(0, 1)
            preprocessed.append(item)
        return preprocessed

    def compute_loss(self, logits, tf_inputs, pad_id):
        """
        Args:
            logits: [len, bsz, vocab]
            tf_inputs: [len, bsz] ("teacher-forced inputs", inputs to decoder used to generate logits)
                (text_indices_w_sos_eos)
        """
        logits = logits[:-1, :, :]    # last input that produced logits is EOS. Don't care about the EOS -> mapping
        targets = tf_inputs[1: :, :]  # remove first input (sos)

        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=pad_id)
        return loss

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        :param batch:
            strokes: [max_stroke_len, bsz, 5] FloatTensor
            stroke_lens: list of ints
            prestrokes: [max_prestrokes_len, bsz, 5] FloatTensor
            prestroke_lens: list of ints
            texts: list of strs
            text_lens: list of ints
            text_indices_w_sos_eos: [max_text_len + 2, bsz] LongTensor (+2 for sos and eos)
            cats: list of strs (categories)
            cats_idx: list of ints
        
        :return: dict: 'loss': float Tensor must exist
        """
        if self.hp.model_type == 'cnn_lstm':
            return self.one_forward_pass_cnn_lstm(batch)
        elif self.hp.model_type == 'transformer_lstm':
            return self.one_forward_pass_transformer_lstm(batch)
        elif self.hp.model_type == 'lstm':
            return self.one_forward_pass_lstm(batch)
        elif self.hp.model_type == 'transformer':
            return self.one_forward_pass_transformer(batch)

    def one_forward_pass_cnn_lstm(self, batch):
        strokes, stroke_lens, prestrokes, prestroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        embedded = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                            category_embedding=self.category_embedding, categories=cats_idx)
        # [bsz, dim]
        embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
        hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
        cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)  # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, dim]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss}

        return result

    def one_forward_pass_transformer_lstm(self, batch):
        strokes, stroke_lens, prestrokes, prestroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        hidden = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                            category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, dim]
        # [bsz, dim]
        hidden = hidden.unsqueeze(0)  # [1, bsz, dim]
        hidden = hidden.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
        cell = hidden.clone()  # [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)  # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, dim]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss}

        return result

    def one_forward_pass_lstm(self, batch):
        strokes, stroke_lens, prestrokes, prestroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        _, (hidden, cell) = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                                     category_embedding=self.category_embedding, categories=cats_idx)
        # [bsz, max_stroke_len, dim]; h/c = [layers * direc, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)      # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, dim]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss}

        return result

    def one_forward_pass_transformer(self, batch):
        strokes, stroke_lens, prestrokes, prestroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Embed strokes and text
        strokes_emb = self.strokes_input_fc(strokes)                   # [max_stroke_len, bsz, dim]
        texts_emb = self.token_embedding(text_indices_w_sos_eos)      # [max_text_len + 2, bsz, dim]

        #
        # Encode decode with transformer
        #
        # Scaling and positional encoding
        enc_inputs = scale_add_pos_emb(strokes_emb, self.pos_enc)
        dec_inputs = scale_add_pos_emb(texts_emb, self.pos_enc)

        src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = \
            create_transformer_padding_masks(stroke_lens, text_lens)
        tgt_mask = generate_square_subsequent_mask(dec_inputs.size(0))  # [max_text_len + 2, max_text_len + 2]
        dec_outputs = self.transformer(enc_inputs, dec_inputs,
                                       src_key_padding_mask=src_key_padding_mask,
                                       # tgt_key_padding_mask=tgt_key_padding_mask, #  TODO: why does adding this result in Nans?
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       tgt_mask=tgt_mask)
        # dec_outputs: [max_text_len + 2, bsz, dim]

        if (dec_outputs != dec_outputs).any():
            import pdb; pdb.set_trace()

        # Compute logits and loss
        logits = torch.matmul(dec_outputs, self.token_embedding.weight.t())  # [max_text_len + 2, bsz, vocab]
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss}

        return result


    # End of epoch hook
    def end_of_epoch_hook(self, epoch, outputs_path=None, writer=None):

        for model in self.models:
            model.eval()

        with torch.no_grad():

            # Generate texts on validation set
            generated = []
            for i, batch in enumerate(self.val_loader):
                batch = self.preprocess_batch_from_data_loader(batch)
                strokes, stroke_lens, prestrokes, prestroke_lens, \
                    texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch
                bsz = strokes.size(1)


                # Model-specific decoding
                if self.hp.model_type in ['cnn_lstm', 'transformer_lstm', 'lstm']:
                    if self.hp.model_type == 'cnn_lstm':
                        # Encode strokes
                        embedded = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                                            category_embedding=self.category_embedding, categories=cats_idx)
                        # [bsz, dim]
                        embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
                        hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
                        cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
                    elif self.hp.model_type == 'transformer_lstm':
                        # Encode strokes
                        hidden = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                                          category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, dim]
                        # [bsz, dim]
                        hidden = hidden.unsqueeze(0)  # [1, bsz, dim]
                        hidden = hidden.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
                        cell = hidden.clone()  # [n_layers, bsz, dim]

                    elif self.hp.model_type == 'lstm':
                        _, (hidden, cell) = self.enc(strokes, stroke_lens,
                                                     prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                                                     category_embedding=self.category_embedding, categories=cats_idx)
                        # [max_stroke_len, bsz, dim]; h/c = [layers * direc, bsz, dim]

                    # Create init input
                    init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
                    init_ids.transpose_(0, 1)  # [1, bsz]

                    decoded_probs, decoded_ids, decoded_texts = self.dec.generate(
                        self.token_embedding,
                        category_embedding=self.category_embedding, categories=cats_idx,
                        init_ids=init_ids, hidden=hidden, cell=cell,
                        pad_id=PAD_ID, eos_id=EOS_ID, max_len=25,
                        decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
                        idx2token=self.tr_loader.dataset.idx2token,
                    )

                elif self.hp.model_type == 'transformer':
                    strokes_emb = self.strokes_input_fc(strokes)  # [max_stroke_len, bsz, dim]
                    src_input_embs = scale_add_pos_emb(strokes_emb, self.pos_enc)  # [max_stroke_len, bsz, dim]

                    init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
                    init_ids.transpose_(0, 1)  # [1, bsz]
                    init_embs = self.token_embedding(init_ids)  # [1, bsz, dim]

                    decoded_probs, decoded_ids, decoded_texts = transformer_generate(
                        self.transformer, self.token_embedding, self.pos_enc,
                        src_input_embs=src_input_embs, input_lens=stroke_lens,
                        init_ids=init_ids,
                        pad_id=PAD_ID, eos_id=EOS_ID,
                        max_len=100,
                        decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
                        idx2token=self.tr_loader.dataset.idx2token)


                for j, instruction in enumerate(texts):
                    generated.append({
                        'ground_truth': instruction,
                        'generated': decoded_texts[j],
                        'url': urls[j],
                        'category': cats[j],
                    })
                    text = 'Ground truth: {}  \n  \nGenerated: {}  \n  \nURL: {}'.format(
                        instruction, decoded_texts[j], urls[j])
                    writer.add_text('inference/sample', text, epoch * self.val_loader.__len__() + j)

            out_fp = os.path.join(outputs_path, 'samples_e{}.json'.format(epoch))
            utils.save_file(generated, out_fp, verbose=True)

            self.compute_metrics_on_generations(generated)

    def compute_metrics_on_generations(self, generated):
        """
        Args:
            generated: list of dicts with 'ground_truth' and 'generated'
        """
        # TODO: compute ROUGE, BLEU
        pass



if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    # Add additional arguments to parser
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'stroke2instruction', run_name)
    utils.save_run_data(save_dir, hp)

    model = StrokeToInstructionModel(hp, save_dir)
    model.train_loop()

    # val_dataset = ProgressionPairDataset('valid')
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
    #                         collate_fn=ProgressionPairDataset.collate_fn)
    # idx2token = val_loader.dataset.idx2token
    # for batch in val_loader:
    #     strokes, stroke_lens, prestrokes, prestroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch
    #     import pdb; pdb.set_trace()
