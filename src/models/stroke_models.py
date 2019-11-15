# stroke_models.py

"""
StrokeDataset and stroke related models
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from src.data_manager.quickdraw import normalize_strokes, stroke3_to_stroke5, build_category_index, final_categories
from src.models.transformer_utils import *

from src.models import nn_utils

NPZ_DATA_PATH = 'data/quickdraw/npz/'

##############################################################################
#
# DATASET
#
##############################################################################

class StrokeDataset(Dataset):
    """
    Dataset to load sketches

    Stroke-3 format: (delta-x, delta-y, binary for if pen is lifted)
    Stroke-5 format: consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
        one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
    """

    def __init__(self, categories, dataset_split, max_len=200):
        """
        Args:
            categories: str (comma separated or 'all')
            dataset_split: str ('train', 'valid', 'test')
            hp: HParams
        """
        self.dataset_split = dataset_split
        self.max_len = max_len

        # get categories
        self.categories = None
        if categories == 'all':
            self.categories = final_categories()
        elif ',' in categories:
            self.categories = categories.split(',')
        else:
            self.categories = [categories]

        # Load data
        full_data = []  # list of dicts
        self.max_len_in_data = 0
        for i, category in enumerate(self.categories):
            print('Loading {} ({}/{})'.format(category, i + 1, len(self.categories)))
            data_path = os.path.join(NPZ_DATA_PATH, '{}.npz'.format(category))
            category_data = np.load(data_path, encoding='latin1')[dataset_split]  # e.g. cat.npz is in 3-stroke format
            n_samples = len(category_data)
            for i in range(n_samples):
                stroke3 = category_data[i]
                sample_len = stroke3.shape[0]  # number of points in stroke3 format
                self.max_len_in_data = max(self.max_len_in_data, sample_len)
                full_data.append({'stroke3': stroke3, 'category': category})

        self.data = self.filter_and_clean_data(full_data)
        self.data = normalize_strokes(self.data)
        self.idx2cat, self.cat2idx = build_category_index(self.data)

        print('Number of examples in {}: {}'.format(dataset_split, len(self.data)))

    def filter_and_clean_data(self, data):
        """
        Removes short and large sequences;
        Remove large gaps (stroke has large delta, i.e. takes place far away from previous stroke)

        Args:
            data: list of dicts
            data: [len, 3 or 5] array (stroke3 or stroke5 format)
        """
        filtered = []
        for sample in data:
            stroke = sample['stroke3']
            stroke_len = stroke.shape[0]
            if (stroke_len > 10) and (stroke_len <= self.max_len):
                # Following means absolute value of offset is at most 1000
                stroke = np.minimum(stroke, 1000)
                stroke = np.maximum(stroke, -1000)
                stroke = np.array(stroke, dtype=np.float32)  # type conversion
                sample['stroke3'] = stroke
                filtered.append(sample)
        return filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        category = sample['category']
        cat_idx = self.cat2idx[category]
        stroke3 = sample['stroke3']
        stroke_len = len(stroke3)
        stroke5 = stroke3_to_stroke5(stroke3, self.max_len_in_data)
        return stroke5, stroke_len, category, cat_idx

        # TODO: do I need to write my own collate_fn like in InstructionGen?

##############################################################################
#
# Encoders
#
##############################################################################

class StrokeEncoderCNN(nn.Module):
    def __init__(self, filter_sizes=[3,4,5], n_feat_maps=128, input_dim=None, emb_dim=None, dropout=None,
                 use_categories=False):
        """
        Args:
            filter_sizes: list of ints
                - Size of convolution window (referred to as filter widths in original paper)
            n_feat_maps: int
                - Number of output feature maps for each filter size
            input_dim: int (size of inputs)
            emb_dim: int (size of embedded inputs)
            dropout_prob: float
            use_categories: bool
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

    def forward(self, strokes, stroke_lens,
                category_embedding=None, categories=None):
        """
        Args:
            strokes: [seq_len, bsz, input_dim]

        Returns:  [bsz, dim]
        """
        # TODO: shouldn't I be using stroke_lens to mask?
        # TODO: implement use_categories

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

class StrokeEncoderLSTM(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True,
                 use_categories=False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_categories = use_categories

        if use_categories:
            self.dropout_mod = nn.Dropout(dropout)
            self.stroke_cat_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True,
                                num_layers=num_layers, dropout=dropout, batch_first=batch_first)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True,
                                num_layers=num_layers, dropout=dropout, batch_first=batch_first)

    def forward(self, strokes, stroke_lens,
                category_embedding=None, categories=None):
        """
        Args:
            strokes:  [max_stroke_len, bsz, input_dim]
            stroke_lens: list of ints, length max_stroke_len
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor

        Returns:
            stroke_outputs: [max_stroke_len, bsz, dim]
            hidden: [layers * direc, bsz, dim]
            cell:  [layers * direc, bsz, dim]
        """
        # Compute a category embedding, repeat it along the time dimension, concatenate it with the strokes along
        # the feature dimension, and apply a fully connected
        bsz = strokes.size(1)

        if self.use_categories:
            cats_emb =  category_embedding(categories)  # [bsz, dim]
            cats_emb = self.dropout_mod(cats_emb)
            strokes = torch.cat([strokes, cats_emb.repeat(strokes.size(0), 1, 1)], dim=2)  # [len, bsz, input+hidden]
            strokes = self.stroke_cat_fc(strokes)  # [len, bsz, hidden]

        packed_strokes = nn.utils.rnn.pack_padded_sequence(strokes, stroke_lens, enforce_sorted=False)
        strokes_outputs, (hidden, cell) = self.lstm(packed_strokes)
        strokes_outputs, _ = nn.utils.rnn.pad_packed_sequence(strokes_outputs)

        # Take mean along num_directions because decoder is unidirectional lstm (this is bidirectional)
        hidden = hidden.view(self.num_layers, 2, bsz, self.hidden_dim).mean(dim=1)  # [layers, bsz, dim]
        cell = cell.view(self.num_layers, 2, bsz, self.hidden_dim).mean(dim=1)  # [layers, bsz, dim]


        return strokes_outputs, (hidden, cell)

class StrokeEncoderTransformer(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim, num_layers=1, dropout=0,
                 use_categories=False
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
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
                category_embedding=None, categories=None):
        """
        Args:
            strokes:  [max_stroke_len, bsz, input_dim]
            stroke_lens: list of ints, length max_stroke_len
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor

        Returns:
            hidden: [bsz, dim]
        """
        bsz = strokes.size(1)

        if self.use_categories:
            cats_emb =  category_embedding(categories)  # [bsz, dim]
            cats_emb = self.dropout_mod(cats_emb)
            strokes = torch.cat([strokes, cats_emb.repeat(strokes.size(0), 1, 1)], dim=2)  # [len, bsz, input+hidden]
            strokes = self.stroke_cat_fc(strokes)  # [len, bsz, hidden]

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

class SketchRNNVAEEncoder(nn.Module):
    def __init__(self, input_dim, enc_dim, enc_num_layers, z_dim, dropout=1.0):
        super().__init__()
        self.enc_dim = enc_dim

        self.lstm = nn.LSTM(input_dim, enc_dim, num_layers=enc_num_layers, dropout=dropout, bidirectional=True)
        # Create mu and sigma by passing lstm's last output into fc layer (Eq. 2)
        self.fc_mu = nn.Linear(2 * enc_dim, z_dim)  # 2 for bidirectional
        self.fc_sigma = nn.Linear(2 * enc_dim, z_dim)

    def forward(self, strokes, hidden_cell=None):
        """
        Args:
            strokes: [max_len, bsz, input_dim] (input_size == isz == 5)
            hidden_cell: tuple of [n_layers * n_directions, bsz, dim]

        Returns:
            z: [bsz, z_dim]
            mu: [bsz, z_dim]
            sigma_hat [bsz, z_dim] (used to calculate KL loss, eq. 10)
        """
        bsz = strokes.size(1)

        # Initialize hidden state and cell state with zeros on first forward pass
        num_directions = 2 if self.lstm.bidirectional else 1
        if hidden_cell is None:
            hidden = torch.zeros(self.lstm.num_layers * num_directions, bsz, self.enc_dim)
            cell = torch.zeros(self.lstm.num_layers * num_directions, bsz, self.enc_dim)
            hidden, cell = nn_utils.move_to_cuda(hidden), nn_utils.move_to_cuda(cell)
            hidden_cell = (hidden, cell)

        # Pass inputs, hidden, and cell into encoder's lstm
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
        _, (hidden, cell) = self.lstm(strokes, hidden_cell)  # h and c: [n_layers * n_directions, bsz, enc_dim]
        # TODO: seems throw a CUDNN error without the float... but shouldn't it be float already?
        last_hidden = hidden.view(self.lstm.num_layers, num_directions, bsz, self.enc_dim)[-1, :, :, :]
        # [num_directions, bsz, hsz]
        last_hidden = last_hidden.transpose(0, 1).reshape(bsz, -1)  # [bsz, num_directions * hsz]

        # Get mu and sigma from hidden
        mu = self.fc_mu(last_hidden)  # [bsz, z_dim]
        sigma_hat = self.fc_sigma(last_hidden)  # [bsz, z_dim]

        if (sigma_hat != sigma_hat).any():
            import pdb;
            pdb.set_trace()
            print('Nans in encoder sigma_hat')

        # Get z for VAE using mu and sigma, N ~ N(0,1)
        # Turn sigma_hat vector into non-negative std parameter
        sigma = torch.exp(sigma_hat / 2.)
        N = torch.randn_like(sigma)
        N = nn_utils.move_to_cuda(N)
        z = mu + sigma * N  # [bsz, z_dim]

        # Note we return sigma_hat, not sigma to be used in KL-loss (eq. 10)
        return z, mu, sigma_hat


##############################################################################
#
# Decoder
#
##############################################################################

class SketchRNNDecoderGMM(nn.Module):
    """
    """

    def __init__(self, input_dim, dec_dim, M, dropout=1.0):
        """
        Args:
            input_dim: int (size of input)
            dec_dim: int (size of hidden states)
            M: int (number of mixtures)
            dropout: float
        """
        super().__init__()

        self.input_dim = input_dim
        self.dec_dim = dec_dim
        self.M = M

        self.lstm = nn.LSTM(input_dim, dec_dim, num_layers=1, dropout=dropout)
        # x_i = [S_{i-1}, z], [h_i; c_i] = forward(x_i, [h_{i-1}; c_{i-1}])     # Eq. 4
        self.fc_params = nn.Linear(dec_dim, 6 * M + 3)  # create mixture params and probs from hiddens

    def forward(self, strokes, output_all=True, hidden_cell=None):
        """
        Args:
            strokes: [len, bsz, esz + 5]
            output_all: boolean, return output at every timestep or just the last
            hidden_cell: tuple of [n_layers, bsz, dec_dim]

        :returns:
            pi: weights for each mixture            [max_len + 1, bsz, M]
            mu_x: mean x for each mixture           [max_len + 1, bsz, M]
            mu_y: mean y for each mixture           [max_len + 1, bsz, M]
            sigma_x: var x for each mixture         [max_len + 1, bsz, M]
            sigma_y: var y for each mixture         [max_len + 1, bsz, M]
            rho_xy:  covariance for each mixture    [max_len + 1, bsz, M]
            q: [max_len + 1, bsz, 3]
                models p (3 pen strokes in stroke-5) as categorical distribution (page 3);   
            hidden: [1, bsz, dec_dim]
                 last hidden state      
            cell: [1, bsz, dec_dim]
                  last cell state
        """
        bsz = strokes.size(1)
        if hidden_cell is None:  # init
            hidden = torch.zeros(self.lstm.num_layers, bsz, self.dec_dim)
            cell = torch.zeros(self.lstm.num_layers, bsz, self.dec_dim)
            hidden, cell = nn_utils.move_to_cuda(hidden), nn_utils.move_to_cuda(cell)
            hidden_cell = (hidden, cell)

        outputs, (hidden, cell) = self.lstm(strokes, hidden_cell)

        # Pass hidden state at each step to fully connected layer (Fig 2, Eq. 4)
        # Dimensions
        #   outputs: [max_len + 1, bsz, dec_dim]
        #   view: [(max_len + 1) * bsz, dec_dim]
        #   y: [(max_len + 1) * bsz, 6 * M + 3] (6 comes from 5 for params, 6th for weights; see page 3)
        if output_all:
            y = self.fc_params(outputs.view(-1, self.dec_dim))
        else:
            y = self.fc_params(hidden.view(-1, self.dec_dim))

        # Separate pen and mixture params
        params = torch.split(y, 6,
                             1)  # splits into tuple along 1st dim; tuple of num_mixture [(max_len + 1) * bsz, 6]'s, 1 [(max_len + 1) * bsz, 3]
        params_mixture = torch.stack(params[:-1])  # trajectories; [num_mixtures, (max_len + 1) * bsz, 6]
        params_pen = params[-1]  # pen up/down;  [(max_len + 1) * bsz, 3]

        # Split trajectories into each mixture param
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # TODO: these all have [num_mix, (max_len+1) * bsz, 1]
        pi = pi.squeeze(2)
        mu_x = mu_x.squeeze(2)
        mu_y = mu_y.squeeze(2)
        sigma_x = sigma_x.squeeze(2)
        sigma_y = sigma_y.squeeze(2)
        rho_xy = rho_xy.squeeze(2)

        # When training, lstm receives whole input, use all outputs from lstm
        # When generating, input is just last generated sample
        # len_out used to reshape mixture param tensors
        if output_all:
            len_out = outputs.size(0)
        else:
            len_out = 1

        # TODO: don't think I actually need the squeeze's
        # if len_out == 1:
        #     import pdb; pdb.set_trace()  # squeeze may be related to generation with len_out = 1
        # TODO: add dimensions
        pi = F.softmax(pi.t().squeeze(), dim=-1).view(len_out, -1, self.M)
        mu_x = mu_x.t().squeeze().contiguous().view(len_out, -1, self.M)
        mu_y = mu_y.t().squeeze().contiguous().view(len_out, -1, self.M)

        # Eq. 6
        sigma_x = torch.exp(sigma_x.t().squeeze()).view(len_out, -1, self.M)
        sigma_y = torch.exp(sigma_y.t().squeeze()).view(len_out, -1, self.M)
        rho_xy = torch.tanh(rho_xy.t().squeeze()).view(len_out, -1, self.M)

        # Eq. 7
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, 3)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell
