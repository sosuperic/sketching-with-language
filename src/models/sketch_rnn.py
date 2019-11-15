# sketch_rnn.py

"""
SketchRNN model is VAE with GMM in decoder
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src import utils
from src.models import nn_utils
from src.data_manager.quickdraw import final_categories, stroke3_to_stroke5, normalize_strokes, \
    build_category_index
from src.models.train_nn import TrainNN, RUNS_PATH

NPZ_DATA_PATH = 'data/quickdraw/npz/'

USE_CUDA = torch.cuda.is_available()


##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        # Data
        self.categories = 'cat'  # comma separated categories or 'all'

        # Training
        self.batch_size = 64  # 100
        self.lr = 0.001  # 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 100

        # Model
        self.model_type = 'vae'  # 'vae', 'decoder'
        self.enc_dim = 256  # 512
        self.dec_dim = 512  # 2048
        self.enc_num_layers = 1  # 2
        self.z_dim = 128  # dimension of z for VAE
        self.M = 20  # number of mixtures
        self.dropout = 0.9  #
        self.wKL = 0.5  # weight for Kl-loss (eq. 11)
        self.temperature = 0.4  # randomness (1=determinstic) in sampling (eq.8)
        self.max_len = 200  # maximum number of strokes in a drawing

        # Annealing -- first focus more on reconstruction loss which is harder to optimize
        # I.e. once L_KL drops low enough,
        # eta_step = 1 - (1 - eta_min) * R
        # loss_train = loss_reconstruction + wKL * eta_step * max(loss_KL, KL_min)
        self.eta_min = 0.01  # initial annealing term(supp eq. 4)
        self.R = 0.99995  # annealing (supp eq. 4)
        self.KL_min = 0.2  #

        self.notes = ''


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
            print('Loading {} ({}/{})'.format(category, i+1, len(self.categories)))
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
# UTILS
#
##############################################################################

def save_sequence_as_img(sequence, output_fp):
    """
    TODO: move to quickdraw?
    
    Args:
        sequence: [len, TODO: is it 3 or 5]
            [delta_x, delta_y, pen up / down]  # TODO: is it up or down
        output_fp: str
    """
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    pil_image.save(output_fp)
    plt.close('all')


##############################################################################
#
# ENCODER
#
##############################################################################
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
# DECODER
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


class SketchRNNModel(TrainNN):
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        self.eta_step = hp.eta_min
        self.tr_loader = self.get_data_loader('train', hp.batch_size, hp.categories, shuffle=True)
        self.val_loader = self.get_data_loader('valid', hp.batch_size, hp.categories, shuffle=False)
        self.end_epoch_loader = self.get_data_loader('train', 1, hp.categories, shuffle=False)

    #
    # Data
    #
    def get_data_loader(self, dataset_split, batch_size, categories, shuffle=True):
        """
        Args:
            dataset_split: str
            batch_size: int
            categories: str
            shuffle: bool
        """
        ds = StrokeDataset(categories, dataset_split, self.hp.max_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        return loader

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

    def preprocess_batch_from_data_loader(self, batch):
        """
        Transposes strokes, moves to cuda

        Args:
            batch: tuple of
                strokes: [bsz, max_len, 5] Tensor
                stroke_lens: list of ints
                cats: list of strs (categories)
                cats_idx: [bsz] LongTensor

        Returns:
            batch: [max_len, bsz, 5]
            stroke_lens: list of ints
        """
        strokes, stroke_lens, cats, cats_idx = batch
        strokes = strokes.transpose(0, 1).float()
        strokes = nn_utils.move_to_cuda(strokes)
        stroke_lens = stroke_lens.numpy().tolist()
        cats_idx = nn_utils.move_to_cuda(cats_idx)
        return strokes, stroke_lens, cats, cats_idx

    #
    # Training, Generation
    #
    def pre_forward_train_hook(self):
        # update eta for LKL
        self.eta_step = 1 - (1 - self.hp.eta_min) * self.hp.R

    def end_of_epoch_hook(self, data_loader, epoch, outputs_path=None, writer=None):  # TODO: is this how to use **kwargs
        self.save_generation(data_loader, epoch, n_gens=1, outputs_path=outputs_path)

    def save_generation(self, gen_data_loader, epoch, n_gens=1, outputs_path=None):
        pass

    #
    # Reconstruction loss
    #
    def reconstruction_loss(self,
                            mask,
                            dx, dy, p,  # ground truth
                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                            q):
        """
        Based on likelihood (Eq. 9)

        Args:
            mask: [max_len + 1, bsz]
            dx: [max_len + 1, bsz, num_mixtures]
            dy: [max_len + 1, bsz, num_mixtures]
            p:  [max_len + 1, bsz, 3]
            pi: [max_len + 1, bsz, M] 

        These are outputs from make_targets(batch, stroke_lens). "+ 1" because of the
        end of sequence stroke appended in make_targets()
        """
        max_len, batch_size = mask.size()

        # Loss w.r.t pen offset
        prob = self.bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        LS = -torch.sum(mask * torch.log(1e-6 + torch.sum(pi * prob, 2))) / float(max_len * batch_size)

        # Loss of pen parameters (cross entropy between ground truth pen params p
        # and predicted categorical distribution q)
        # LP = -torch.sum(p * torch.log(q)) / float(max_len * batch_size)
        LP = F.binary_cross_entropy(q, p, reduction='mean')  # Maybe this gets read of NaN?
        #  TODO: check arguments for above BCE

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
        norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
        exp = torch.exp(-z / (2 * (1 - rho_xy ** 2)))

        return exp / norm


class SketchRNNDecoderOnlyModel(SketchRNNModel):
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        # Model
        self.dec = SketchRNNDecoderGMM(5, hp.dec_dim, hp.M)
        self.models.append(self.dec)
        if USE_CUDA:
            for model in self.models:
                model.cuda()

        # optimization -- ADAM plus annealing (supp eq. 4)
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        Args:
            batch: tuple from DataLoaders

        Returns:
            dict where 'loss': float Tensor must exist
        """
        strokes, stroke_lens, cats, cats_idx = batch
        max_len, bsz, _ = strokes.size()

        # Create inputs to decoder
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)  # start of sequence
        sos = nn_utils.move_to_cuda(sos)
        dec_inputs = torch.cat([sos, strokes], 0)  # add sos at the begining of the strokes; [max_len + 1, bsz, 5]

        # Decode
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)

        # Calculate losses
        mask, dx, dy, p = self.make_target(strokes, stroke_lens, self.hp.M)
        loss = self.reconstruction_loss(mask,
                                        dx, dy, p,
                                        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                        q)
        result = {'loss': loss, 'loss_R': loss}

        return result

    def save_generation(self, gen_data_loader, epoch, n_gens=1, outputs_path=None):
        # TODO: generation not implemented yet (need to generalize the conditional generation of the VAE)
        pass


class SketchRNNVAEModel(SketchRNNModel):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """

    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        # Model
        self.enc = SketchRNNVAEEncoder(5, hp.enc_dim, hp.enc_num_layers, hp.z_dim, dropout=hp.dropout)
        self.fc_hc = nn.Linear(hp.z_dim, 2 * hp.dec_dim)  # 2: 1 for hidden, 1 for cell
        self.dec = SketchRNNDecoderGMM(hp.z_dim + 5, hp.dec_dim, hp.M)
        self.models.extend([self.enc, self.fc_hc, self.dec])
        if USE_CUDA:
            for model in self.models:
                model.cuda()

        # optimization -- ADAM plus annealing (supp eq. 4)
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        Args:
            batch: tuple from loader

        Returns:
            dict where 'loss': float Tensor must exist
        """
        strokes, stroke_lens, cats, cats_idx = batch
        max_len, bsz, _ = strokes.size()

        # Encode
        z, mu, sigma_hat = self.enc(strokes)  # each [bsz, z_dim]

        # Create inputs to decoder
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)  # start of sequence
        sos = nn_utils.move_to_cuda(sos)
        inputs_init = torch.cat([sos, strokes], 0)  # add sos at the begining of the strokes; [max_len + 1, bsz, 5]
        z_stack = torch.stack([z] * (max_len + 1), dim=0)  # expand z to concat with inputs; [max_len + 1, bsz, z_dim]
        dec_inputs = torch.cat([inputs_init, z_stack], 2)  # each input is stroke + z; [max_len + 1, bsz, z_dim + 5]

        # init hidden and cell states is tanh(fc(z)) (Page 3)
        hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_dim, 1)
        hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        # TODO: if we want multiple layers, we need to replicate hidden and cell n_layers times

        # Decode
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True,
                                                                     hidden_cell=hidden_cell)

        # Calculate losses
        mask, dx, dy, p = self.make_target(strokes, stroke_lens, self.hp.M)
        loss_KL = self.kullback_leibler_loss(sigma_hat, mu, self.hp.KL_min, self.hp.wKL, self.eta_step)
        loss_R = self.reconstruction_loss(mask,
                                          dx, dy, p,
                                          pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                          q)
        loss = loss_KL + loss_R
        result = {'loss': loss, 'loss_KL': loss_KL, 'loss_R': loss_R}

        return result

    #
    # KL loss
    #
    def kullback_leibler_loss(self, sigma_hat, mu, KL_min, wKL, eta_step):
        """
        Calculate KL loss -- (eq. 10, 11)

        Args:
            sigma_hat: [bsz, z_dim]
            mu: [bsz, z_dim]
            KL_min: float
            wKL: float
            eta_step: float

        Returns: float Tensor
        """
        bsz, z_dim = sigma_hat.size()

        LKL = -0.5 * torch.sum(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat)) \
              / float(bsz * z_dim)
        KL_min = torch.Tensor([KL_min])
        KL_min = nn_utils.move_to_cuda(KL_min)
        KL_min = KL_min.detach()

        LKL = wKL * eta_step * torch.max(LKL, KL_min)
        return LKL

    ##############################################################################
    # Conditional generation
    ##############################################################################
    def save_generation(self, data_loader, epoch, n_gens=1, outputs_path=None):
        """
        Generate sequence conditioned on output of encoder
        """
        n = 0
        for i, batch in enumerate(data_loader):
            batch = self.preprocess_batch_from_data_loader(batch)
            strokes, stroke_lens, cats, cats_idx = batch

            max_len, bsz, _ = strokes.size()

            # Encode
            z, _, _ = self.enc(strokes)  # z: [1, 1, 128]  # TODO: is z actually [1, 128]?

            # initialize state with start of sequence stroke-5 stroke
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            sos = nn_utils.move_to_cuda(sos)

            # generate until end of sequence or maximum sequence length
            s = sos
            seq_x = []  # delta-x
            seq_y = []  # delta-y
            seq_pen = []  # pen-down
            # init hidden and cell states is tanh(fc(z)) (Page 3)
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_dim, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
            for _ in range(max_len):
                input = torch.cat([s, z.unsqueeze(0)], 2)  # [1,1,133]

                # decode
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = \
                    self.dec(input, output_all=False, hidden_cell=hidden_cell)
                hidden_cell = (hidden, cell)

                # sample next state
                s, dx, dy, pen_down, eos = self.sample_next_state(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
                seq_x.append(dx)
                seq_y.append(dy)
                seq_pen.append(pen_down)

                if eos:  # done drawing
                    break

            # get in format to draw image
            # Cumulative sum because seq_x and seq_y are deltas, so get x (or y) at each stroke
            sample_x = np.cumsum(seq_x, 0)
            sample_y = np.cumsum(seq_y, 0)
            sample_pen = np.array(seq_pen)
            sequence = np.stack([sample_x, sample_y, sample_pen]).T
            output_fp = os.path.join(outputs_path, '{}-{}.jpg'.format(epoch, n))
            save_sequence_as_img(sequence, output_fp)

            n += 1
            if n == n_gens:
                break

    def sample_next_state(self, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
        """
        Return state using current mixture parameters etc. set from decoder call.
        Note that this state is different from the stroke-5 format.

        Args:
            pi: [len, bsz, M]
            # TODO!! This is currently hardcoded with save_conditional_generation
            # When we do pi.data[0,0,:] down below,
                The 0-th index 0: Currently, pi is being calculated with output_all=False, which means it's just ouputting the last pi.
                The 1-th index 0: first in batch
        # TODO: refactor so that the above isn't the case.

        Returns:
            # TODO: what does it return exactly?
            s, dx, dy, pen_down, eos
        """
        M = pi.size(-1)

        def adjust_temp(pi_pdf):
            """Not super sure why this instead of just dividing by temperauture as in eq. 8, but
            magenta sketch_run/model.py does it this way(adjust_temp())"""
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # Get mixture index
        # pi is weights for mixtures
        pi = pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)

        pi_idx = np.random.choice(M, p=pi)  # choose Gaussian weighted by pi

        # Get mixture params
        mu_x = mu_x.data[0, 0, pi_idx]
        mu_y = mu_y.data[0, 0, pi_idx]
        sigma_x = sigma_x.data[0, 0, pi_idx]
        sigma_y = sigma_y.data[0, 0, pi_idx]
        rho_xy = rho_xy.data[0, 0, pi_idx]

        # Get next x andy by using mixture params and sampling from bivariate normal
        dx, dy = self.sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)

        # Get pen state
        q = q.data[0, 0, :].cpu().numpy()
        q_idx = np.random.choice(3, p=q)

        # Create next_state vector
        next_state = torch.zeros(5)
        next_state[0] = dx
        next_state[1] = dy
        next_state[q_idx + 2] = 1
        next_state = nn_utils.move_to_cuda(next_state)
        s = next_state.view(1, 1, -1)

        pen_down = q_idx == 1  # TODO: isn't this pen up?
        eos = q_idx == 2

        return s, dx, dy, pen_down, eos

    def sample_bivariate_normal(self, mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False):
        """
        """
        if greedy:
            return mu_x, mu_y

        mean = [mu_x, mu_y]

        # randomness controlled by temperature (eq. 8)
        sigma_x *= np.sqrt(self.hp.temperature)
        sigma_y *= np.sqrt(self.hp.temperature)

        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
               [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]

        # TODO: is this right
        mean = [v.item() for v in mean]
        for i, row in enumerate(cov):
            cov[i] = [v.item() for v in row]
        x = np.random.multivariate_normal(mean, cov, 1)

        return x[0][0], x[0][1]


if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'sketchrnn', run_name)
    utils.save_run_data(save_dir, hp)

    model = None
    if hp.model_type == 'vae':
        model = SketchRNNVAEModel(hp, save_dir)

    elif hp.model_type == 'decoder':
        model = SketchRNNDecoderOnlyModel(hp, save_dir)
    model = nn_utils.AccessibleDataParallel(model)
    model.train_loop()
