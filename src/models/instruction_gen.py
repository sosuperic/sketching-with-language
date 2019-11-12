# instruction_gen.py

import argparse
import matplotlib
matplotlib.use('Agg')
from nltk.tokenize import word_tokenize  # TODO: add the download punkt to requirements.txt
import numpy as np
import os

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
        self.max_epochs = 100

        # Model
        self.dim = 256
        self.n_enc_layers = 4
        self.n_dec_layers = 4
        self.model_type = 'lstm'  # 'lstm', 'transformer'
        self.condition_on_hc = True  # With 'lstm', input to decoder also contains last hidden cell
        self.use_prestrokes = True
        self.use_categories = False
        self.dropout = 0.2

        # inference
        self.decode_method = 'greedy'  # 'sample', 'greedy'
        self.tau = 1.0  # sampling text
        self.k = 5      # sampling text



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

class StrokeEncoderLSTM(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True,
                 use_prestrokes=False,
                 n_categories=0,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim,
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_prestrokes = use_prestrokes
        self.n_categories = n_categories

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers, dropout=dropout, batch_first=batch_first)
        if n_categories > 0:
            self.category_embedding = nn.Embedding(n_categories, hidden_dim)

    def forward(self, strokes, stroke_lens, prestrokes=None, prestroke_lens=None, categories=None):
        """
        Args:
            strokes:  [max_stroke_len, bsz]
            stroke_lens: list of ints, length max_stroke_len
            prestrokes:  [max_prestroke_len, bsz]
            prestroke_lens: list of ints, length max_prestroke_len
            categories: [bsz] LongTensor

        Returns:
            stroke_outputs: [max_stroke_len, bsz, dim]
            prestroke_outputs: [max_prestroke_len, bsz, dim]
            hidden: [layers * direc, bsz, dim]
            cell:  [bsz, max_stroke_len, dim]
        """
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

        if self.n_categories > 0:
            cats_emb =  self.category_embedding(categories)  # [bsz, dim]
            hidden += cats_emb.unsqueeze(0)

        return strokes_outputs, (hidden, cell)

class InstructionDecoderLSTM(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True,
                 condition_on_hc=False, use_prestrokes=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim,
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.condition_on_hc = condition_on_hc
        self.use_prestrokes = use_prestrokes  # currently not used. only in STrokeEncoderLSTM

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers, dropout=dropout, batch_first= batch_first)

    def forward(self, texts_emb, text_lens, hidden=None, cell=None, token_embedding=None):
        """
        Args:
            texts_emb: [len, bsz, dim] FloatTensor
            text_lens: list of ints, length len
            hidden: [n_layers * n_directions, bsz, dim]  FloatTensor
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
            token_embedding: nn.Embedding(vocab, dim)
            
        Returns:
            outputs:
                if token_embedding is None: [len, bsz, dim] FloatTensor 
                else: [len, bsz, vocab] FloatTensor 
            hidden: [n_layers * n_directions, bsz, dim]
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
        """

        # Condition on last layer's hidden and cell on every time step
        if self.condition_on_hc:
            # combine last hidden and cell, repeat along time dimension, and concatenate with encoded texts
            last_hidden, last_cell = hidden[-1, :, :], cell[-1, :, :]  # last = [bsz, dim]
            last_hc = (last_hidden + last_cell).unsqueeze(0)  # [1, bsz, dim]
            last_hc = last_hc.repeat(texts_emb.size(0), 1, 1)  # [len, bsz, dim]
            inputs_emb = torch.cat([texts_emb, last_hc], dim=2)  # [len, bsz, dim * 2]
        else:
            inputs_emb = texts_emb

        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs_emb, text_lens, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed_inputs, (hidden, cell))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # [max_text_len + 2, bsz, dim]; h/c = [n_layers * n_directions, bsz, dim]

        if token_embedding is not None:
            outputs = torch.matmul(outputs, token_embedding.weight.t())  # [len, bsz, vocab]

        return outputs, (hidden, cell)

    def generate(self, token_embedding,
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
        rows_with_eos = nn_utils.move_to_cuda(torch.zeros(bsz).long())
        pad_ids = nn_utils.move_to_cuda(torch.Tensor(bsz).fill_(pad_id)).long()
        pad_prob = nn_utils.move_to_cuda(torch.zeros(bsz, vocab_size))  # one hot for pad id
        pad_prob[:, pad_id] = 1

        # Generate
        decoded_probs = nn_utils.move_to_cuda(torch.zeros(bsz, max_len, vocab_size))  #
        decoded_ids = nn_utils.move_to_cuda(torch.zeros(bsz, max_len).long())  # [bsz, max_len]
        cur_input_id = init_ids
        for t in range(max_len):
            cur_input_emb = token_embedding(cur_input_id)  # [1, bsz, dim]
            if self.condition_on_hc:
                last_hc = hidden[-1, :, :] + cell[-1, :, :]  # [bsz, dim]
                last_hc = last_hc.unsqueeze(0)  # [1, bsz, dim]
                cur_input_emb = torch.cat([cur_input_emb, last_hc], dim=2)  # [1, bsz, dim * 2]
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

        # import pdb; pdb.set_trace()

        return decoded_probs, decoded_ids, decoded_texts

class StrokeToInstructionModel(TrainNN):
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        self.tr_loader, self.val_loader = self.get_data_loaders()

        # Model
        if self.hp.model_type == 'transformer':
            self.strokes_input_fc = nn.Linear(5, self.hp.dim)
            self.pos_enc = PositionalEncoder(self.hp.dim, max_seq_len=250)
            self.transformer = nn.Transformer(
                d_model=self.hp.dim, dim_feedforward=self.hp.dim * 4, nhead=2, activation='relu',
                num_encoder_layers=self.hp.n_enc_layers, num_decoder_layers=self.hp.n_dec_layers,
                dropout=hp.dropout,
            )
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.models.extend([self.strokes_input_fc, self.pos_enc, self.transformer])

            if self.hp.use_prestrokes:
                print('Using prestrokes not implemented for Transformer')
                # raise NotImplementedError('Using prestrokes not implemented for Transformer')

        elif self.hp.model_type == 'lstm':
            n_categories = 35 if self.hp.use_categories else 0
            self.enc = StrokeEncoderLSTM(
                5, self.hp.dim, num_layers=self.hp.n_enc_layers,
                dropout=self.hp.dropout, batch_first=False,
                use_prestrokes=self.hp.use_prestrokes,
                n_categories=n_categories,
            )

            dec_input_dim = self.hp.dim
            if self.hp.condition_on_hc:
                dec_input_dim += self.hp.dim
            self.dec = InstructionDecoderLSTM(
                dec_input_dim, self.hp.dim, num_layers=self.hp.n_dec_layers,
                dropout=self.hp.dropout, batch_first=False,
                condition_on_hc=self.hp.condition_on_hc, use_prestrokes=self.hp.use_prestrokes)

            self.models.extend([self.enc, self.dec])

        self.token_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, self.hp.dim)
        self.models.extend([self.token_embedding])

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
        if self.hp.model_type == 'transformer':
            return self.one_forward_pass_transformer(batch)
        elif self.hp.model_type == 'lstm':
            return self.one_forward_pass_lstm(batch)

    def one_forward_pass_lstm(self, batch):
        strokes, stroke_lens, prestrokes, prestroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        _, (hidden, cell) = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                                     categories=cats_idx)
        # [bsz, max_stroke_len, dim]; h/c = [layers * direc, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)      # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell, token_embedding=self.token_embedding)  # [max_text_len + 2, bsz, dim]; h/c
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



    #
    # End of epoch hook
    #
    def end_of_epoch_hook(self, epoch, outputs_path=None, writer=None):
        if self.hp.model_type == 'transformer':
            self.end_of_epoch_hook_transformer(epoch, outputs_path, writer)
        elif self.hp.model_type == 'lstm':
            self.end_of_epoch_hook_lstm(epoch, outputs_path, writer)

    def end_of_epoch_hook_lstm(self, epoch, outputs_path=None, writer=None):
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

                # Encode

                _, (hidden, cell) = self.enc(strokes, stroke_lens, prestrokes=prestrokes, prestroke_lens=prestroke_lens,
                                             categories=cats_idx)
                # [max_stroke_len, bsz, dim]; h/c = [layers * direc, bsz, dim]

                # Create init input
                init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
                init_ids.transpose_(0,1)  # [1, bsz]

                decoded_probs, decoded_ids, decoded_texts = self.dec.generate(
                    self.token_embedding,
                    init_ids=init_ids, hidden=hidden, cell=cell,
                    pad_id=PAD_ID, eos_id=EOS_ID, max_len=25,
                    decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
                    idx2token=self.tr_loader.dataset.idx2token,
                    )

                for j, instruction in enumerate(texts):
                    generated.append({
                        'ground_truth': instruction,
                        'generated': decoded_texts[j],
                        'url': urls[j]
                    })
                    text = 'Ground truth: {}  \n  \nGenerated: {}  \n  \nURL: {}'.format(
                        instruction, decoded_texts[j], urls[j])
                    writer.add_text('inference/sample', text, epoch * self.val_loader.__len__() + j)

            out_fp = os.path.join(outputs_path, 'samples_e{}.json'.format(epoch))
            utils.save_file(generated, out_fp, verbose=True)


    def end_of_epoch_hook_transformer(self, epoch, outputs_path=None, writer=None):

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

                strokes_emb = self.strokes_input_fc(strokes)                    # [max_stroke_len, bsz, dim]
                src_input_embs = scale_add_pos_emb(strokes_emb, self.pos_enc)    # [max_stroke_len, bsz, dim]

                init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
                init_ids.transpose_(0,1)  # [1, bsz]
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
                        'url': urls[j]
                    })
                    text = 'Ground truth: {}  \n  \nGenerated: {}  \n  \nURL: {}'.format(
                        instruction, decoded_texts[j], urls[j])
                    writer.add_text('inference/sample', text, epoch * self.val_loader.__len__() + j)

            out_fp = os.path.join(outputs_path, 'samples_e{}.json'.format(epoch))
            utils.save_file(generated, out_fp, verbose=True)


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
