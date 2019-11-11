

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

from data_manager.quickdraw import LABELED_PROGRESSION_PAIRS_PATH, LABELED_PROGRESSION_PAIRS_DATA_PATH
from models.sketch_rnn import normalize_data, calculate_normalizing_scale_factor, stroke3_to_stroke5, TrainNN
from utils import save, load


LABELED_PROGRESSION_PAIRS_TRAIN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'train.pkl')
LABELED_PROGRESSION_PAIRS_VALID_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'valid.pkl')
LABELED_PROGRESSION_PAIRS_TEST_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'test.pkl')

LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2token.json')
LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'token2idx.json')
LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2cat.json')
LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'cat2idx.json')


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
        self.n_enc_layers = 4  # 2
        self.n_dec_layers = 4  # 2
        # dropout


##############################################################################
#
# DATASET
#
##############################################################################

PAD, OOV, EOS, SOS = 0, 1, 2, 3  # TODO: this should be a part of dataset maybe?

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
    tokens = [OOV, EOS, SOS] + list(tokens)
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
        data = load(fp)
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
        save(data, fp)

    # build and save vocab
    idx2token, token2idx = build_vocab(train + valid + test)
    for data, fp in [(idx2token, LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH),
                     (token2idx, LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)]:
        save(data, fp)

    # build and save category to index map (in case our model conditions on category)
    idx2cat, cat2idx = build_category_index(train + valid + test)
    for data, fp, in [(idx2cat, LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH),
                      (cat2idx, LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)]:
        save(data, fp)

def map_str_to_index(s, token2idx):
    return [token2idx[tok] for tok in normalize(s)]


def normalize_data(data):
    """
    Normalize entire dataset (delta_x, delta_y) by the scaling factor.

    :param: data is list of dicts
    """
    scale_factor = calculate_normalizing_scale_factor(data)
    normalized = []
    for seq in data:
        seq[:, 0:2] /= scale_factor
        normalized.append(seq)
    return normalized


class ProgressionPairDataset(Dataset):
    def __init__(self, dataset_split):
        """
        
        :param split_data: list of dicts
        """
        super().__init__()

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
        data = load(fp)

        # Load vocab and category mappings
        self.idx2token = load(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.token2idx = load(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.vocab_size = len(self.idx2token)

        self.idx2cat = load(LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH)
        self.cat2idx = load(LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)

        # TODO: need to modify normalize_data because data in sketch_rnn is stroke-5 format,
        # here split_data is a list of dicts with stroke3_
        # TODO: Do we use the normalizing scale factor for the entire sketch dataset or calculated just on this annotated data?
        # self.data = normalize_data(data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        stroke3_seg = sample['stroke3_segment']

        # TODO: why do we even need stroke5 format
        stroke5_seg = stroke3_to_stroke5(stroke3_seg, len(stroke3_seg))
        # TODO: is just passing the length okay for max_len? we'll do the batching manually

        text = sample['annotation']
        text_indices = map_str_to_index(text, self.token2idx)

        cat = sample['category']
        cat_idx = self.cat2idx[cat]

        return (stroke5_seg, text, text_indices, cat, cat_idx)

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch
        
        :param: batch: list of samples, one sample is returned from __getitem__(idx)
        """
        strokes, texts, texts_indices, cats, cats_idx = zip(*batch)
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

        return batch_strokes, stroke_lens, texts, text_lens, batch_text_indices, cats, cats_idx




##############################################################################
#
# Transformers
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
#
##############################################################################

import math
class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=250):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] =  math.sin(pos / (10000 ** ((2 * i) / d_model)))
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

def create_transformer_padding_masks(src_lens, tgt_lens):
    """
    Return ByteTensors where a true value means value should be ignored. Used to handle variable length
    sequences within a batch.
    
    :param src_lens: list of length bsz
    :param tgt_lens: list of length bsz
    :return:
        src_key_padding_mask: [bsz, max_src_len] ByteTensor
        tgt_key_padding_mask: [bsz, max_tgt_len] ByteTensor
        memory_key_padding_mask: [bsz, max_src_len] ByteTensor
    """
    bsz = len(src_lens)
    max_src_len = max(src_lens)
    src_key_padding_mask = torch.zeros(bsz, max_src_len).bool()
    for i, seq_len in enumerate(src_lens):
        src_key_padding_mask[i,seq_len:] = 1

    max_tgt_len = max(tgt_lens)
    tgt_key_padding_mask = torch.zeros(bsz, max_tgt_len).bool()
    for i, seq_len in enumerate(tgt_lens):
        tgt_key_padding_mask[i,seq_len:] = 1

    memory_key_padding_mask = src_key_padding_mask

    if USE_CUDA:
        src_key_padding_mask = src_key_padding_mask.cuda()
        tgt_key_padding_mask = tgt_key_padding_mask.cuda()
        memory_key_padding_mask = memory_key_padding_mask.cuda()

    return src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

def generate_square_subsequent_mask(size):
    """
    Generate a square mask for the sequence that prevents attending to items in the future.
    
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    if USE_CUDA:
        mask = mask.cuda()

    return mask



##############################################################################
#
# MODEL
#
##############################################################################


class InstructionRNN(TrainNN):
    def __init__(self, hp):
        super().__init__(hp)

        self.tr_loader, self.val_loader = self.get_data_loaders()

        # Model
        d_model = self.hp.dim 
        
        self.pos_enc = PositionalEncoder(d_model, max_seq_len=250)  # [1, max_seq_len, dim]  (1 for broadcasting with bsz)
        self.strokes_input_fc = nn.Linear(5, d_model)
        self.tokens_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, d_model)
        self.enc_dec = nn.Transformer(d_model=d_model, dim_feedforward=d_model * 4,
                                     nhead=2,
                                     activation='gelu',
                                     num_encoder_layers=self.hp.n_enc_layers,
                                     num_decoder_layers=self.hp.n_dec_layers)
        self.vocab_out_fc = nn.Linear(d_model, self.tr_loader.dataset.vocab_size)
        if USE_CUDA:
            self.pos_enc.cuda()
            self.strokes_input_fc.cuda()
            self.tokens_embedding.cuda()
            self.enc_dec.cuda()
            self.vocab_out_fc.cuda()

        # Optimizezrs
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
        """Convert tensors to cuda"""
        strokes, stroke_lens, texts, text_lens, text_indices, cats, cats_idx = batch
        if USE_CUDA:
            strokes = strokes.cuda()
            text_indices = text_indices.cuda()
        batch = (strokes, stroke_lens, texts, text_lens, text_indices, cats, cats_idx)
        return batch

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        :param batch:
            strokes: [bsz, max_stroke_len, 3 or 5] FloatTensor
            stroke_lens: list of ints
            texts: list of strs
            text_lens: list of ints
            text_indices_w_sos_eos: [bsz, max_text_len + 2] LongTensor (+2 for SOS and EOS)
            cats: list of strs (categories)
            cats_idx: list of ints
        
        :return: dict: 'loss': float Tensor must exist
        """
        strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx = batch
        bsz = strokes.size(0)

        # Embed strokes and text
        stroke_emb = self.strokes_input_fc(strokes)                   # [bsz, max_stroke_len, dim]
        text_emb = self.tokens_embedding(text_indices_w_sos_eos)      # [bsz, max_text_len + 2, dim]

        #
        # Encode decode with transformer
        #
        # Scaling and positional encoding
        enc_inputs = stroke_emb * math.sqrt(self.hp.dim)
        dec_inputs = text_emb * math.sqrt(self.hp.dim)
        enc_inputs += self.pos_enc.pe[:,:enc_inputs.size(1),:]
        dec_inputs += self.pos_enc.pe[:,:dec_inputs.size(1),:]

        # transpose because transformer expects length dimension first
        enc_inputs.transpose_(0, 1)  # [max_stroke_len, bsz, dim]
        dec_inputs.transpose_(0, 1)  # [max_text_len + 2, bsz, dim]

        src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = \
            create_transformer_padding_masks(stroke_lens, text_lens)
        tgt_mask = generate_square_subsequent_mask(dec_inputs.size(0))  # [max_text_len + 2, max_text_len + 2]

        dec_outputs = self.enc_dec(enc_inputs, dec_inputs,  # [max_text_len + 2, bsz, dim]
                                   src_key_padding_mask=src_key_padding_mask,
                                   # tgt_key_padding_mask=tgt_key_padding_mask,
                                   #  TODO: why does adding this result in Nans? Technically don't need it anyway though probably given that we have tgt_mask?
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_mask=tgt_mask
                                   )
        # src_mask and memory_mask: don't think there should be one

        #
        # Compute logits and loss
        #
        logits = self.vocab_out_fc(dec_outputs)  # [max_text_len + 2, bsz, vocab]

        logits = logits.transpose(0,1)  # [bsz, max_text_len + 2, vocab]
        logits = logits[:,:-1,:]  # Last input is EOS, output would be EOS -> <token>. Should be ignored.
        vocab_size = logits.size(-1)
        text_indices_w_eos = text_indices_w_sos_eos[:,1:]  # remove sos; [bsz, max_text_len + 1]

        loss = F.cross_entropy(logits.reshape(-1, vocab_size),  # [bsz * max_text_len + 1, vocab]
                               text_indices_w_eos.reshape(-1),  # [bsz * max_text_len + 1]
                               ignore_index=PAD)  # TODO: is ignore_index enough for masking loss value?

        result = {'loss': loss}

        return result




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', type=str, default='instruction')
    args = parser.parse_args()

    hp = HParams()
    model = InstructionRNN(hp)
    model.train_loop(args.model_name)