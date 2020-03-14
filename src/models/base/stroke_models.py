# stroke_models.py

"""
StrokeDataset(s) and stroke related models
"""

import json
import numpy as np
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision

import haste_pytorch as haste

from config import SEGMENTATIONS_PATH, NPZ_DATA_PATH
from src.data_manager.quickdraw import normalize_strokes, stroke3_to_stroke5, build_category_index, final_categories, \
    ndjson_drawings, ndjson_to_stroke3
from src.models.core.transformer_utils import *
from src.models.core import nn_utils
# from src.models.core.custom_lstms import script_lnlstm, LSTMState
from src.models.core.layernormlstm import LayerNormLSTM
import src.models.core.resnet_with_cbam as resnet_with_cbam

##############################################################################
#
# DATASET
#
##############################################################################

# This was precomputed once on the training set of all 35 categories
STROKE3_SCALE_FACTOR = 41.712997


class StrokeDataset(Dataset):
    """
    Returns drawing in stroke3 or stroke5 format
    """
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

    def __getitem__(self, idx, pad_to_max_len_in_data=True):
        sample = self.data[idx]
        category = sample['category']
        cat_idx = self.cat2idx[category]
        stroke3 = sample['stroke3']
        stroke_len = len(stroke3)

        # data augmentation
        if self.data_aug:
            noise = np.random.uniform(0.9, 1.1, size=(stroke3.shape[0], 2))
            stroke3[:,:2] *= noise

        if pad_to_max_len_in_data:
            stroke5 = stroke3_to_stroke5(stroke3, self.max_len_in_data)
            # TODO: padding to the max in data is a bit non-transparent / unnecessary imo
            # Maybe I should write a collate_fn to pad to max in batch.
        else:
            stroke5 = stroke3_to_stroke5(stroke3)

        stroke_len += 1  # during stroke 5 conversion, there is a end of drawing point
        return stroke5, stroke_len, category, cat_idx

class NdjsonStrokeDataset(StrokeDataset):
    """
    Load from the simplified_ndjson files
    """
    def __init__(self, categories, dataset_split, max_len=200, max_per_category=70000, must_have_instruction_tree=False):
        self.categories = categories
        self.dataset_split = dataset_split
        self.max_len = max_len
        self.max_per_category = max_per_category
        self.must_have_instruction_tree = must_have_instruction_tree
        self.data_aug = dataset_split == 'train'

        if categories == 'all':
            self.categories = final_categories()
        elif ',' in categories:
            self.categories = categories.split(',')
        else:
            self.categories = [categories]

        # Load data
        full_data = []  # list of dicts
        self.max_len_in_data = 0
        n_cats = len(self.categories)

        start1 = time.time()
        for i, category in enumerate(self.categories):
            # print(f'Loading {category} {i+1}/{n_cats}')
            drawing_lines = ndjson_drawings(category, lazy=True)  # each line is a json of one drawing
            drawing_lines = self.get_split(drawing_lines, dataset_split)  # filter to subset for this split

            cat_n_drawings = 0
            for line in drawing_lines:
                # load drawing
                d = json.loads(line)
                if not d['recognized']:
                    continue

                if cat_n_drawings == max_per_category:
                    break
                id, ndjson_strokes = d['key_id'], d['drawing']

                if must_have_instruction_tree:  # example must have an instruction tree generated
                    # TODO: hardcoded... should be moved
                    instruction_tree_fp = SEGMENTATIONS_PATH / 'greedy_parsing' / 'ndjson' / 'nov30_2019' / 'strokes_to_instruction' /  category / f'{id}.json'
                    if not os.path.exists(instruction_tree_fp):
                        continue

                stroke3 = ndjson_to_stroke3(ndjson_strokes)  # convert to stroke3 format
                sample_len = stroke3.shape[0] + 1  # + 1 because stroke5 appends 1 basically
                self.max_len_in_data = max(self.max_len_in_data, sample_len)
                full_data.append({'stroke3': stroke3, 'category': category,
                                  'id': id, 'ndjson_strokes': ndjson_strokes})
                cat_n_drawings += 1

        print('Loading: ', time.time() - start1)

        start2 = time.time()
        self.data = self.filter_and_clean_data(full_data)
        print('Filter and clean: ', time.time() - start2)

        start3 = time.time()
        self.data = normalize_strokes(self.data)
        print('Normalize: ', time.time() - start2)
        self.idx2cat, self.cat2idx = build_category_index(self.data)

        print(f'Number of examples in {dataset_split}: {len(self.data)}')

    def get_split(self, drawing_lines, dataset_split):
        """
        Given list of ndjson data, returns subset corresponding to that split
        """
        tr_amt, val_amt, te_amt = 0.9, 0.05, 0.05

        l = len(drawing_lines)
        tr_idx = int(tr_amt * l)
        val_idx = int((tr_amt + val_amt) * l)

        if dataset_split == 'train':
            return drawing_lines[:tr_idx]
        elif dataset_split == 'valid':
            return drawing_lines[tr_idx:val_idx]
        elif dataset_split == 'test':
            return drawing_lines[val_idx:]


class NpzStrokeDataset(StrokeDataset):
    """
    Load from the npz files
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
        self.data_aug = dataset_split == 'train'

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
            data_path = NPZ_DATA_PATH / f'{category}.npz'
            category_data = np.load(data_path, encoding='latin1')[dataset_split]  # e.g. cat.npz is in 3-stroke format
            n_samples = len(category_data)
            for i in range(n_samples):
                stroke3 = category_data[i]
                sample_len = stroke3.shape[0] + 1 # + 1 because stroke5 appends 1 basically
                self.max_len_in_data = max(self.max_len_in_data, sample_len)
                full_data.append({'stroke3': stroke3, 'category': category})

        self.data = self.filter_and_clean_data(full_data)
        self.data = normalize_strokes(self.data, scale_factor=STROKE3_SCALE_FACTOR)
        self.idx2cat, self.cat2idx = build_category_index(self.data)

        print(f'Number of examples in {dataset_split}: {len(self.data)}')


##############################################################################
#
# Standard Encoders
#
##############################################################################

class StrokeAsImageEncoderCNN(nn.Module):
    def __init__(self, cnn_type, n_channels, output_dim):
        """

        Args:
            cnn_type (str): wideresnet, cbam, or se
            n_channels (int): Number of input channels, which can vary depending on if
                pre, post, full images are used
            output_dim (int): output dim
        """
        super().__init__()
        self.cnn_type = cnn_type
        self.n_channels = n_channels
        self.output_dim = output_dim

        # Load cnn and override first conv to take appropriate number of channels
        if cnn_type == 'wideresnet':
            self.cnn = torchvision.models.wide_resnet50_2()
            self.cnn.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.cnn.fc = nn.Linear(2048, output_dim)
        elif cnn_type == 'se':
            self.cnn = torch.hub.load('moskomule/senet.pytorch', 'se_resnet50', pretrained=False)
            self.cnn.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.cnn.fc = nn.Linear(2048, output_dim)
        elif cnn_type == 'cbam':
            # Using CIFAR and not ImageNet because ImageNet model expects larger images (224 works, 112 doesn't)
            self.cnn = resnet_with_cbam.ResidualNet('CIFAR100', 50, output_dim, 'CBAM')
            self.cnn.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # self.cnn.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


    def forward(self, images):
        """
        Args:
            images: [n_channels, bsz, W, H]

        Returns:  [bsz, dim]
        """
        images = images.transpose(0,1)  # [B, C, W, H]
        encoded = self.cnn(images)
        return encoded

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
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=False,
                 use_categories=False, use_layer_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.use_categories = use_categories
        self.use_layer_norm = use_layer_norm

        if use_categories:
            self.dropout_mod = nn.Dropout(dropout)
            self.stroke_cat_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
            if use_layer_norm:
                self.lstm = LayerNormLSTM(hidden_dim, hidden_dim, bidirectional=True,
                                    num_layers=num_layers)  # only batch second, no dropout
            else:
                self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True,
                                    num_layers=num_layers, dropout=dropout, batch_first=batch_first)
        else:
            if use_layer_norm:
                self.lstm = LayerNormLSTM(input_dim, hidden_dim, bidirectional=True,
                                    num_layers=num_layers)  # only batch second, no dropout
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
            strokes_outputs: [max_stroke_len, bsz, num_directions (2) * dim]
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

        if self.use_layer_norm:  # torchscript doesn't support packed sequence
            init_states = (nn_utils.move_to_cuda(torch.zeros(self.num_layers * 2, bsz, self.hidden_dim)),
                           nn_utils.move_to_cuda(torch.zeros(self.num_layers * 2, bsz, self.hidden_dim)))
            strokes_outputs, (hidden_, cell_) = self.lstm(strokes, init_states)  # layernorm lstm must pass in states
            # TODO: is this the proper way to mask / account for lengths?
            # At least in the strokes_to_instruction case, we just care about the last hidden states
            # Note: the following is a little different from the non-layernorm version. In that case,
            # 1) hiddens and cells are distinct
            # 2) hidden and cells at each layer are returned (here it's repeated, outputs is only the last layer)
            hidden = [strokes_outputs[stroke_lens[i]-1, i, :] for i in range(bsz)]  # bsz length list, items are [dim]
            hidden = torch.stack(hidden, dim=0)  # [bsz, dim]
            hidden = hidden.repeat(self.num_layers, 1, 1)  # [num_layers, bsz, dim]
            cell = hidden.clone()

        else:
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


##############################################################################
#
# VAE Encoder
#
##############################################################################

class SketchRNNVAEEncoder(nn.Module):
    def __init__(self, input_dim, enc_dim, enc_num_layers, z_dim, dropout=0.0):
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
# GMM Decoder
#
##############################################################################

class SketchRNNDecoderGMM(nn.Module):
    def __init__(self, input_dim, dec_dim, M, dropout=0.0,
                       use_layer_norm=False, rec_dropout=0.0):
        """
        Args:
            input_dim: int (size of input)
            dec_dim: int (size of hidden states)
            M: int (number of mixtures)
            use_layer_norm (bool)
            rec_dropout: float
                - https://arxiv.org/pdf/1603.05118.pdf
        """
        super().__init__()

        self.input_dim = input_dim
        self.dec_dim = dec_dim
        self.M = M
        self.num_layers = 1
        self.use_layer_norm = use_layer_norm

        if use_layer_norm:
            self.lstm = haste.LayerNormLSTM(input_size=input_dim, hidden_size=dec_dim, zoneout=dropout, dropout=rec_dropout)
            # self.lstm = LayerNormLSTM(input_dim, dec_dim, num_layers=self.num_layers, rec_dropout=rec_dropout)
        else:
            self.lstm = nn.LSTM(input_dim, dec_dim, num_layers=self.num_layers, dropout=dropout)
        # x_i = [S_{i-1}, z], [h_i; c_i] = forward(x_i, [h_{i-1}; c_{i-1}])     # Eq. 4
        self.fc_params = nn.Linear(dec_dim, 6 * M + 3)  # create mixture params and probs from hiddens

    def forward(self, strokes, stroke_lens=None, output_all=True, hidden_cell=None):
        """
        Args:
            strokes: [len, bsz, input_dim (e.g. dim + 5)]
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


        if self.use_layer_norm:
            outputs, (hidden, cell) = self.lstm(strokes)
        else:
            if hidden_cell is None:  # init
                hidden = torch.zeros(self.lstm.num_layers, bsz, self.dec_dim)
                cell = torch.zeros(self.lstm.num_layers, bsz, self.dec_dim)
                hidden, cell = nn_utils.move_to_cuda(hidden), nn_utils.move_to_cuda(cell)
                hidden_cell = (hidden, cell)
            outputs, (hidden, cell) = self.lstm(strokes, hidden_cell)
        # self.outputs = outputs

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
        params = torch.split(y, 6, dim=1)  # tuple of M [(max_len + 1) * bsz, 6] tensors, 1 [(max_len + 1) * bsz, 3] tensor
        params_mixture = torch.stack(params[:-1])  # trajectories; [M, (max_len + 1) * bsz, 6]
        params_pen = params[-1]  # pen up/down;  [(max_len + 1) * bsz, 3]

        # Split trajectories into each mixture param
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, dim=2)
        # These all have [M, (max_len+1) * bsz, 1]; squeeze
        pi = pi.squeeze(2)  # [M, (max_len+1) * bsz]
        mu_x = mu_x.squeeze(2)  # [M, (max_len+1) * bsz]
        mu_y = mu_y.squeeze(2)  # [M, (max_len+1) * bsz]
        sigma_x = sigma_x.squeeze(2)  # [M, (max_len+1) * bsz]
        sigma_y = sigma_y.squeeze(2)  # [M, (max_len+1) * bsz]
        rho_xy = rho_xy.squeeze(2)  # [M, (max_len+1) * bsz]

        # When training, lstm receives whole input, use all outputs from lstm
        # When generating, input is just last generated sample
        # len_out used to reshape mixture param tensors
        if output_all:
            len_out = outputs.size(0)
        else:
            len_out = 1

        # Compute softmax over mixtures
        pi = F.softmax(pi.t(), dim=-1).view(len_out, bsz, self.M)

        mu_x = mu_x.t().contiguous().view(len_out, bsz, self.M)
        mu_y = mu_y.t().contiguous().view(len_out, bsz, self.M)

        # Eq. 6
        sigma_x = torch.exp(sigma_x.t()).view(len_out, bsz, self.M)
        sigma_y = torch.exp(sigma_y.t()).view(len_out, bsz, self.M)
        rho_xy = torch.tanh(rho_xy.t()).view(len_out, bsz, self.M)

        # Eq. 7
        q = F.softmax(params_pen, dim=-1).view(len_out, bsz, 3)

        # TODO: refactor all instances to unpack outputs
        # TODO: rename outputs as all_hidden?
        return outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

    #
    # Loss
    #
    def make_target(self, strokes, stroke_lens, M):
        """
        Create target vector out of stroke-5 data and stroke_lens. Namely, use stroke_lens
        to create mask for each sequence

        Args:
            strokes: [max_len, bsz, 5]
            stroke_lens: list of ints
            M: int, number of mixtures

        Returns:
            mask: [max_len + 1, bsz]
            dx: [max_len + 1, bsz, num_mixtures]
            dy: [max_len + 1, bsz, num_mixtures]
            p:  [max_len + 1, bsz, 3]
        """
        max_len, bsz, _ = strokes.size()

        # add eos
        eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * bsz).unsqueeze(0)  # ([1, bsz, 5])
        eos = nn_utils.move_to_cuda(eos)
        strokes = torch.cat([strokes, eos], 0)  # [max_len + 1, bsz, 5]

        # calculate mask for each sequence using stroke_lens
        mask = torch.zeros(max_len + 1, bsz)
        for idx, length in enumerate(stroke_lens):
            mask[:length, idx] = 1
        mask = nn_utils.move_to_cuda(mask)
        mask = mask.detach()
        dx = torch.stack([strokes.data[:, :, 0]] * M, 2).detach()
        dy = torch.stack([strokes.data[:, :, 1]] * M, 2).detach()
        p1 = strokes.data[:, :, 2].detach()
        p2 = strokes.data[:, :, 3].detach()
        p3 = strokes.data[:, :, 4].detach()
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p

    #
    # Reconstruction loss
    #
    def reconstruction_loss(self,
                            mask,
                            dx, dy, p,  # ground truth
                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                            q,
                            average_loss=True):
        """
        Based on likelihood (Eq. 9)

        Args:
            mask: [max_len + 1, bsz]
            dx: [max_len + 1, bsz, M]
            dy: [max_len + 1, bsz, M]
            p:  [max_len + 1, bsz, 3]
            pi: [max_len + 1, bsz, M]
            mu_x: [max_len + 1, bsz, M]
            mu_y: [max_len + 1, bsz, M]
            sigma_x: [max_len + 1, bsz, M]
            sigma_y: [max_len + 1, bsz, M]
            rho_xy: [max_len + 1, bsz, M]
            q: [max_len + 1, bsz, 3]
            average_loss (bool): whether to average loss per batch item
                When True, returns [bsz] FloatTensor

        These are outputs from make_targets(batch, stroke_lens). "+ 1" because of the
        end of sequence stroke appended in make_targets()
        """
        max_len, batch_size = mask.size()

        # Loss w.r.t pen offset
        prob = self.bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        if average_loss:
            LS = -torch.sum(mask * torch.log(1e-6 + torch.sum(pi * prob, 2))) / mask.sum()
        else:
            LS = -torch.sum(mask * torch.log(1e-6 + torch.sum(pi * prob, 2)), dim=0) / mask.sum(dim=0)  # [bsz]

        if ((LS != LS).any() or (LS == float('inf')).any() or (LS == float('-inf')).any()):
            raise Exception('Nan in SketchRNNDecoderGMM reconstruction loss')

        # Loss of pen parameters (cross entropy between ground truth pen params p
        # and predicted categorical distribution q)
        if average_loss:
            # LP = -torch.sum(p * torch.log(q)) / mask.sum()
            LP = -torch.sum(mask.unsqueeze(-1) * p * torch.log(q)) / mask.sum()
        else:
            # LP = p * torch.log(q)  # [max_len + 1, bsz, 3]
            LP = mask.unsqueeze(-1) * p * torch.log(q)  # [max_len + 1, bsz, 3]
            LP = LP.sum(dim=[0,2]) / mask.sum(dim=0)    # [bsz]

        return LS + LP

    def bivariate_normal_pdf(self, dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
        """
        Get probability of dx, dy using mixture parameters.

        Reference: Eq. of https://arxiv.org/pdf/1308.0850.pdf (Graves' Generating Sequences with
        Recurrent Neural Networks)
        """
        # Eq. 25
        # Reminder: mu's here are calculated for mixture model on the stroke data, which
        # models delta-x's and delta-y's. So z_x just comparing actual ground truth delta (dx)
        # to the prediction from the mixture model (mu_x). Then normalizing etc.
        z_x = ((dx - mu_x) / sigma_x) ** 2
        z_y = ((dy - mu_y) / sigma_y) ** 2
        z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
        z = z_x + z_y - 2 * rho_xy * z_xy

        # Eq. 24
        exp_denom = (2 * (1 - rho_xy ** 2))
        exp = torch.exp(-z / (exp_denom + 1e-5))
        norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
        prob = exp / (norm + 1e-10)

        if (prob != prob).any():
            raise Exception('Nan in SketchRNNDecoderGMM bivariate_normal_pdf')

        return prob

##############################################################################
#
# LSTM Decoder
#
##############################################################################

class SketchRNNDecoderLSTM(nn.Module):
    def __init__(self, input_dim, dec_dim, dropout=0.0):
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
        self.num_layers = 1

        self.lstm = nn.LSTM(input_dim, dec_dim, num_layers=self.num_layers, dropout=dropout)
        # x_i = [S_{i-1}, z], [h_i; c_i] = forward(x_i, [h_{i-1}; c_{i-1}])     # Eq. 4
        self.fc_out_xypen = nn.Linear(dec_dim, 5)  # create mixture params and probs from hiddens

    def forward(self, strokes, stroke_lens=None, hidden_cell=None, output_all=True):
        """
        Args:
            strokes: [max_len + 1, bsz, input_dim]  (+ 1 for sos)
            stroke_lens: list of ints, length len
            hidden_cell: tuple of [n_layers, bsz, dec_dim]
            output_all: bool (unused... for compatability with SketchRNNDecoderGMM)

        Returns:
            xy: [max_len + 1, bsz, 5] (+1 for sos)
            q: [max_len + 1, bsz, 3]
            # xy: [len, bsz, 5] (len may be less than max_len + 1)
            # q: [len, bsz, 3]
                models p (3 pen strokes in stroke-5) as categorical distribution (page 3)
            hidden: [n_layers, bsz, dim]
            cell: [n_layers, bsz, dim]
        """
        bsz = strokes.size(1)
        if hidden_cell is None:  # init
            hidden = torch.zeros(self.lstm.num_layers, bsz, self.dec_dim)
            cell = torch.zeros(self.lstm.num_layers, bsz, self.dec_dim)
            hidden, cell = nn_utils.move_to_cuda(hidden), nn_utils.move_to_cuda(cell)
            hidden_cell = (hidden, cell)

        # decode
        outputs, (hidden, cell) = self.lstm(strokes, hidden_cell)

        # packed_inputs = nn.utils.rnn.pack_padded_sequence(strokes, stroke_lens, enforce_sorted=False)
        # outputs, (hidden, cell) = self.lstm(packed_inputs, (hidden, cell))
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # # [len, bsz, dim]; h/c = [n_layers * n_directions, bsz, dim]
        # # NOTE: pad_packed will "trim" extra timesteps, so outputs may be shorter than strokes

        outputs = self.fc_out_xypen(outputs)  # [len, bsz, 5]
        xy = outputs[:,:,:2]  # [len, bsz, 2]
        pen = outputs[:,:,2:]  # [len, bsz, 3]
        q = F.softmax(pen, dim=-1)

        return xy, q, hidden, cell

    def make_target(self, strokes, stroke_lens):
        """
        Create target vector out of stroke-5 data and stroke_lens. Namely, use stroke_lens
        to create mask for each sequence

        Args:
            strokes: [max_len, bsz, 5]
            stroke_lens: list of ints

        Returns:
            mask: [max_len + 1, bsz]  (+ 1 for eos)
            dxdy: [max_len + 1, bsz, 2]
            p:  [max_len + 1, bsz, 3]
        """
        max_len, bsz, _ = strokes.size()

        # add eos
        eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * bsz).unsqueeze(0)  # ([1, bsz, 5])
        eos = nn_utils.move_to_cuda(eos)
        strokes = torch.cat([strokes, eos], 0)  # [max_len + 1, bsz, 5]

        # calculate mask for each sequence using stroke_lens
        mask = torch.zeros(max_len + 1, bsz)
        for idx, length in enumerate(stroke_lens):
            mask[:length, idx] = 1
        mask = nn_utils.move_to_cuda(mask)
        mask = mask.detach()

        # ignore first element
        dxdy = strokes.data[:, :, :2].detach()
        p1 = strokes.data[:, :, 2].detach()
        p2 = strokes.data[:, :, 3].detach()
        p3 = strokes.data[:, :, 4].detach()
        p = torch.stack([p1, p2, p3], dim=2)

        return mask, dxdy, p

    def reconstruction_loss(self, mask, dxdy, p, xy, q):
        """
        Args:
            mask (FloatTensor): [max_len + 1, bsz]
            dxdy (FloatTensor): [max_len + 1, bsz, 2] (target)
            p (FloatTensor): [max_len + 1, bsz, 3] (target)
            xy (FloatTensor): [max_len + 1, bsz, 2] (model output) (len may be less than max_len + 1)
            q (FloatTensor): [max_len + , bsz, 3] (model output)
            # xy (FloatTensor): [len, bsz, 2] (model output) (len may be less than max_len + 1)
            # q (FloatTensor): [len, bsz, 3] (model output)
        """
        dxdy *= mask.unsqueeze(-1)
        p *= mask.unsqueeze(-1)

        # gen_len, bsz, _ = xy.size()
        # dxdy = dxdy[:gen_len,:,:]
        # p = p[:gen_len,:,:]

        LS = F.mse_loss(xy, dxdy)
        LP = F.mse_loss(q, p)
        loss = LS + LP
        return loss
