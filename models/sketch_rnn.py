# sketch_rnn.py

"""
SketchRNN model is VAE with GMM in decoder
"""

import argparse
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


NPZ_DATA_PATH = 'data/quickdraw/npz/'
RUNS_PATH = 'runs/'

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
        self.lr = 0.001  # 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 100

        # Model
        self.enc_hidden_size = 256 # 512
        self.dec_hidden_size = 512 # 2048
        self.enc_num_layers = 1  # 2
        self.Nz = 128  # dimension of z for VAE
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


##############################################################################
#
# DATASET
#
##############################################################################
class SketchDataset(Dataset):
    """
    Dataset to load sketches

    Stroke-3 format: (delta-x, delta-y, binary for if pen is lifted)
    Stroke-5 format: consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
        one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
    """
    def __init__(self, category, dataset_split, hp):
        """
        
        :param category: str ('cat', 'giraffe', etc.)
        :param dataset_split: str, ('train', 'valid', 'test')
        """
        self.hp = hp

        data_path = os.path.join(NPZ_DATA_PATH, '{}.npz'.format(category))
        full_data = np.load(data_path, encoding='latin1')[dataset_split]  # e.g. cat.npz is in 3-stroke format
        self.data = self._preprocess_data(full_data)
        self.max_len = max([len(seq[:, 0]) for seq in self.data])  # this may be less than the hp max len

    def _preprocess_data(self, data):  # see preprocess() in sketch_rnn/utils.py
        """
        Filter, clean, normalize data
        """
        # Filter first so that normalizing scale factor doesn't use filtered out sequences
        preprocessed = self._filter_and_clean_data(data)
        preprocessed = self._normalize_data(preprocessed)
        preprocessed = self._filter_to_multiple_of_batch_size(preprocessed)
        return preprocessed

    def _filter_and_clean_data(self, data):
        """
        Removes short and large sequences;
        Remove large gaps (stroke has large delta, i.e. takes place far away from previous stroke)
        """
        filtered = []
        for seq in data:
            seq_len = len(seq[:, 0])
            if (seq_len > 10) and (seq_len <= self.hp.max_len):
                # Following means absolute value of offset is at most 1000
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype=np.float32)
                filtered.append(seq)
        return filtered

    def _normalize_data(self, data):
        """
        Normalize entire dataset (delta_x, delta_y) by the scaling factor.
        """
        scale_factor = self._calculate_normalizing_scale_factor(data)
        normalized = []
        for seq in data:
            seq[:, 0:2] /= scale_factor
            normalized.append(seq)
        return normalized

    def _calculate_normalizing_scale_factor(self, data):  # calculate_normalizing_scale_factor() in sketch_rnn/utils.py
        """
        Calculate the normalizing factor in Appendix of paper
        """
        delta_data = []
        for i in range(len(data)):
            for j in range(len(data[i])):
                delta_data.append(data[i][j, 0])
                delta_data.append(data[i][j, 1])
        delta_data = np.array(delta_data)
        scale_factor = np.std(delta_data)
        return scale_factor

    def _filter_to_multiple_of_batch_size(self, data):
        """
        Code requires fixed batch size for some reason
        """
        n_batches = len(data) // self.hp.batch_size
        filtered = data[:n_batches * self.hp.batch_size]
        return filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample, l = self._stroke3_to_stroke5(sample)

        return (sample, l)

    def _stroke3_to_stroke5(self, seq):  # to_big_strokes() in sketch_rnn/utils.py
        """
        Convert from stroke-3 to stroke-5 format 

        :returns
            result: [max_len, 5] float array
            l: int, length of sequence
        """
        result = np.zeros((self.max_len, 5), dtype=float)
        l = len(seq)
        assert l <= self.max_len
        result[0:l, 0:2] = seq[:, 0:2]  # 1st and 2nd values are same
        result[0:l, 3] = seq[:, 2]  # stroke-5[3] = pen-up, same as stroke-3[2]
        result[0:l, 2] = 1 - result[0:l, 3]  # stroke-5[2] = pen-down, stroke-3[2] = pen-up (so inverse)
        result[l:, 4] = 1  # last "stroke" has stroke5[4] equal to 1, all other values 0 (see Figure 4); hence l
        return result, l


##############################################################################
#
# UTILS
#
##############################################################################

def save_sequence_as_img(sequence, output_fp):
    """

    :param sequence: [delta_x, delta_y, pen up / down]  # TODO: is it up or down
    :param output_fp: str
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
class EncoderRNN(nn.Module):
    """
    Outputs:
        z: fed to each step of decoder;                   [batch_size, Nz]
        mu: used to calculate KL loss (eq. 10);           [batch_size, Nz]
        sigma_hat: used to calculate KL loss (eq. 10);    [batch_size, Nz]
    """

    def __init__(self, hp):
        super(EncoderRNN, self).__init__()
        self.hp = hp
        self.lstm = nn.LSTM(5, self.hp.enc_hidden_size,
                            num_layers=self.hp.enc_num_layers, dropout=self.hp.dropout, bidirectional=True)

        # Create mu and sigma by passing lstm's last output into fc layer (Eq. 2)
        self.fc_mu = nn.Linear(2 * self.hp.enc_hidden_size, self.hp.Nz)  # 2 for bidirectional
        self.fc_sigma = nn.Linear(2 * self.hp.enc_hidden_size, self.hp.Nz)

    def forward(self, inputs, hidden_cell=None):
        """

        :param inputs: [max_len, bsz, isz] (input_size == isz == 5)
        :param batch_size: int
        :param hidden_cell:
        
        :return:
            z: [bsz, Nz]
            mu: [bsz, Nz]
            sigma_hat [bsz, Nz]
        """
        bsz = inputs.size(1)

        # Initialize hidden state and cell state with zeros on first forward pass
        num_directions = 2 if self.lstm.bidirectional else 1
        if hidden_cell is None:
            hidden = torch.zeros(self.lstm.num_layers * num_directions, bsz, self.hp.enc_hidden_size)
            cell = torch.zeros(self.lstm.num_layers * num_directions, bsz, self.hp.enc_hidden_size)
            if USE_CUDA:
                hidden = hidden.cuda()
                cell = cell.cuda()
            hidden_cell = (hidden, cell)

        # Pass inputs, hidden, and cell into encoder's lstm
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell) # h and c: [n_layers * n_directions, bsz, hsz]
        last_hidden = hidden.view(self.lstm.num_layers, num_directions, bsz, self.hp.enc_hidden_size)[-1,:,:,:]
        # [num_directions, bsz, hsz]
        last_hidden = last_hidden.transpose(0, 1).reshape(bsz, -1)  # [bsz, num_directions * hsz]

        # Get mu and sigma from hidden
        mu = self.fc_mu(last_hidden)  # [bsz, Nz]
        sigma_hat = self.fc_sigma(last_hidden)  # [bsz, Nz]

        if (sigma_hat != sigma_hat).any():
            import pdb; pdb.set_trace()
            print('Nans in encoder sigma_hat')

        # Get z for VAE using mu and sigma, N ~ N(0,1)
        # Turn sigma_hat vector into non-negative std parameter
        sigma = torch.exp(sigma_hat / 2.)
        N = torch.randn_like(sigma)
        if USE_CUDA:
            N = N.cuda()
        z = mu + sigma * N  # [bsz, Nz]

        # Note we return sigma_hat, not sigma to be used in KL-loss (eq. 10)
        return z, mu, sigma_hat

##############################################################################
#
# DECODER
#
##############################################################################
class DecoderRNN(nn.Module):
    """
    """

    def __init__(self, hp):
        super(DecoderRNN, self).__init__()

        self.hp = hp

        # Initialize decoder states
        # Page 3: The initial hidden states h0, and optional cell states c0 (if applicable)
        # of the decoder RNN is the output of a single layer network [h0; c0] = tanh(W*z + b)
        self.fc_hc = nn.Linear(self.hp.Nz, 2 * self.hp.dec_hidden_size)  # 2: 1 for hidden, 1 for cell

        # Plus 5 for
        # TODO: is this the right equation / shoudl this comment be here
        # x_i = [S_{i-1}, z], [h_i; c_i] = forward(x_i, [h_{i-1}; c_{i-1}])     # Eq. 4
        self.lstm = nn.LSTM(self.hp.Nz + 5, self.hp.dec_hidden_size,
                            num_layers=1, dropout=self.hp.dropout)
                            # num_layers=self.hp.num_layers, dropout=self.hp.dropout)

        # Create mixture params and probs from hiddens
        self.fc_params = nn.Linear(self.hp.dec_hidden_size, 6 * self.hp.M + 3)

    def forward(self, inputs, z, output_all=True, hidden_cell=None):
        """

        :param inputs: [len, bsz, esz + 5]
        :param z: [bsz, esz]
        :param output_all: boolean, return output at every timestep or just the last
        :param hidden_cell: 
        
        :returns:
            pi: weights for each mixture            [max_len + 1, bsz, M]
            mu_x: mean x for each mixture           [max_len + 1, bsz, M]
            mu_y: mean y for each mixture           [max_len + 1, bsz, M]
            sigma_x: var x for each mixture         [max_len + 1, bsz, M]
            sigma_y: var y for each mixture         [max_len + 1, bsz, M]
            rho_xy:  covariance for each mixture    [max_len + 1, bsz, M]
            q: [max_len + 1, bsz, 3]
                models p (3 pen strokes in stroke-5) as categorical distribution (page 3);   
            hidden: [1, bsz, dec_hidden_size]
                 last hidden state      
            cell: [1, bsz, dec_hidden_size]
                  last cell state
        """
        # On first forward pass, initialize hidden state and cell state using z
        if hidden_cell is None:
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), self.hp.dec_hidden_size, 1)
            # TODO: if we want multiple layers, we need to replicate hidden and cell n_layers times
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())

        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)

        # Pass hidden state at each step to fully connected layer (Fig 2, Eq. 4)
        # Dimensions
        #   outputs: [max_len + 1, bsz, dec_hsz]
        #   view: [(max_len + 1) * bsz, dec_hsz]
        #   y: [(max_len + 1) * bsz, 6 * M + 3] (6 comes from 5 for params, 6th for weights; see page 3)
        if output_all:
            y = self.fc_params(outputs.view(-1, self.hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, self.hp.dec_hidden_size))

        # Separate pen and mixture params
        params = torch.split(y, 6, 1)  # splits into tuple along 1st dim; tuple of num_mixture [(max_len + 1) * bsz, 6]'s, 1 [(max_len + 1) * bsz, 3]
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
        pi = F.softmax(pi.t().squeeze(), dim=-1).view(len_out, -1, self.hp.M)
        mu_x = mu_x.t().squeeze().contiguous().view(len_out, -1, self.hp.M)
        mu_y = mu_y.t().squeeze().contiguous().view(len_out, -1, self.hp.M)

        # Eq. 6
        sigma_x = torch.exp(sigma_x.t().squeeze()).view(len_out, -1, self.hp.M)
        sigma_y = torch.exp(sigma_y.t().squeeze()).view(len_out, -1, self.hp.M)
        rho_xy = torch.tanh(rho_xy.t().squeeze()).view(len_out, -1, self.hp.M)

        # Eq. 7
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, 3)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell


##############################################################################
#
# MODEL
#
##############################################################################

class SketchRNN(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

    def make_target(self, batch, lengths, M):
        """
        Create target vector out of stroke-5 data and lengths. Namely, use lengths
        to create mask for each sequence

        :param: batch: [max_len, bsz, 5]
        :param: lengths: list of ints 
        :param: M: int, number of mixtures
        
        :returns: 
            mask: [max_len + 1, bsz]
            dx: [max_len + 1, bsz, num_mixtures]
            dy: [max_len + 1, bsz, num_mixtures]
            p:  [max_len + 1, bsz, 3]
        """
        max_len, bsz, _ = batch.size()

        # add eos
        eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * bsz).unsqueeze(0)  # ([1, bsz, 5])
        if USE_CUDA:
            eos = eos.cuda()
        batch = torch.cat([batch, eos], 0)  # [max_len + 1, bsz, 5]

        # calculate mask for each sequence using lengths
        mask = torch.zeros(max_len + 1, bsz)
        for idx, length in enumerate(lengths):
            mask[:length, idx] = 1
        if USE_CUDA:
            mask = mask.cuda()
        mask = mask.detach()
        dx = torch.stack([batch.data[:, :, 0]] * M, 2).detach()
        dy = torch.stack([batch.data[:, :, 1]] * M, 2).detach()
        p1 = batch.data[:, :, 2].detach()
        p2 = batch.data[:, :, 3].detach()
        p3 = batch.data[:, :, 4].detach()
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p

    def preprocess_batch_from_data_loader(self, batch, lengths):
        """
        :param: batch TODO
        :param: lengths TODO
        
        :returns:
            batch: [max_len, bsz, 5]
            lengths: list of ints
        """
        batch.transpose_(0, 1).float()  # Dataloader returns batch in 1st dimension, rest of code expects it to be 2nd
        if USE_CUDA:
            batch = batch.cuda()
        lengths = lengths.numpy().tolist()
        return batch, lengths

    def lr_decay(self, optimizer, min_lr, lr_decay):
        """
        Decay learning rate by a factor of lr_decay
        """
        for param_group in optimizer.param_groups:
            if param_group['lr'] > min_lr:
                param_group['lr'] *= lr_decay
        return optimizer

    #
    # Reconstruction loss
    #
    def reconstruction_loss(self,
                            mask,
                            dx, dy, p,  # ground truth
                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                            q):
        """
        Eq. 9

        Params
        ------
        These are outputs from make_targets(batch, lengths)
            mask: [max_len + 1, bsz]
            dx: [max_len + 1, bsz, num_mixtures]
            dy: [max_len + 1, bsz, num_mixtures]
            p:  [max_len + 1, bsz, 3]
            pi: [max_len + 1, bsz, M] 
        + 1 because end of sequence stroke appended in make_targets()
        """
        max_len, batch_size = mask.size()

        # Loss w.r.t pen offset
        prob = self.bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(pi * prob, 2))) / float(max_len * batch_size)

        # Loss of pen parameters (cross entropy between ground truth pen params p
        # and predicted categorical distribution q)
        LP = -torch.sum(p * torch.log(q)) / float(max_len * batch_size)

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


class SketchRNNDecoderOnly(SketchRNN):
    def __init__(self, hp):
        super().__init__(hp)

        # Model
        self.decoder = DecoderRNN(hp)
        if USE_CUDA:
            self.decoder.cuda()

        # optimization -- ADAM plus annealing (supp eq. 4)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

class SketchRNNVAE(SketchRNN):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """

    def __init__(self, hp):
        super().__init__(hp)

        # Model
        self.encoder = EncoderRNN(hp)
        self.decoder = DecoderRNN(hp)
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

        # optimization -- ADAM plus annealing (supp eq. 4)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

    def dataset_loop(self, data_loader, epoch, is_train=True, writer=None, tb_tag='train'):
        """Can be used with either different splits of the dataset (train, valid, test)"""

        losses = []
        LRs = []
        LKLs = []
        for i, (batch, lengths) in enumerate(data_loader):
            # Set up optimizers
            if is_train:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                self.eta_step = 1 - (1 - self.hp.eta_min) * self.hp.R  # update eta for LKL

            batch, lengths = self.preprocess_batch_from_data_loader(batch, lengths)  # [max_len, bsz, 5];
            max_len, bsz, _ = batch.size()

            # Encode
            z, mu, sigma_hat = self.encoder(batch)  # each [bsz, Nz]

            # Create inputs to decoder. First, create start of sequence
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)
            if USE_CUDA:
                sos = sos.cuda()

            batch = batch.float()  # TODO: is this right? Also move to preprocess_batch()
            batch_init = torch.cat([sos, batch],
                                   0)  # add sos at the begining of the batch; [max_len + 1, bsz, 5]
            z_stack = torch.stack([z] * (max_len + 1), dim=0)  # expand z to concat with inputs; [max_len + 1, bsz, Nz]
            inputs = torch.cat([batch_init, z_stack], 2)  # each input is stroke + z; [max_len + 1, bsz, Nz + 5]

            # Decode
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.decoder(inputs, z, output_all=True)

            # Get targets
            mask, dx, dy, p = self.make_target(batch, lengths, self.hp.M)

            # Calculate losses
            LKL = self.kullback_leibler_loss(sigma_hat, mu, self.hp.KL_min, self.hp.wKL, self.eta_step)
            LR = self.reconstruction_loss(mask,
                                          dx, dy, p,
                                          pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                          q)
            loss = LR + LKL
            losses.append(loss.item())
            LKLs.append(LKL.item())
            LRs.append(LR.item())

            # Gradient and optimization
            if is_train:
                loss.backward()
                nn.utils.clip_grad_value_(self.encoder.parameters(), self.hp.grad_clip)
                nn.utils.clip_grad_value_(self.decoder.parameters(), self.hp.grad_clip)
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

            # Logging
            if i % 10 == 0:
                step = epoch * data_loader.__len__() + i
                writer.add_scalar('{}/loss'.format(tb_tag), loss.item(), step)
                writer.add_scalar('{}/KL_loss'.format(tb_tag), LKL.item(), step)
                writer.add_scalar('{}/reconstruction_loss'.format(tb_tag), LR.item(), step)

        mean_loss = np.mean(losses)
        mean_LKL = np.mean(LKLs)
        mean_LR = np.mean(LRs)

        return mean_loss, mean_LKL, mean_LR

    def train_loop(self, model_name, category):
        """Train and validate on multiple epochs"""

        # Bookkeeping
        model_name += '_' + category
        run_datetime = datetime.now().strftime('%B%d_%H-%M-%S')
        run_path = os.path.join(RUNS_PATH, model_name, run_datetime)
        print(run_path)
        tb_path = os.path.join(run_path, 'tensorboard')
        output_imgs_path = os.path.join(run_path, 'output_imgs')
        os.makedirs(output_imgs_path)
        writer = SummaryWriter(tb_path)
        stdout_fp = os.path.join(run_path, 'stdout.txt')
        stdout_f = open(stdout_fp, 'w')

        # Get data
        tr_dataset = SketchDataset(category, 'train', self.hp)
        val_dataset = SketchDataset(category, 'valid', self.hp)
        tr_loader = DataLoader(tr_dataset, batch_size=hp.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False)

        cond_gen_data_loader = DataLoader(tr_dataset, batch_size=1, shuffle=False)  # currently cond gen hard coded for bsz = 1
        # cond_gen_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # currently cond gen hard coded for bsz = 1

        # Train
        val_losses = []  # used for early stopping
        min_val_loss = float('inf')  # used to save model
        for epoch in range(self.hp.max_epochs):

            # train
            start_time = time.time()
            self.encoder.train()
            self.decoder.train()
            loss, LKL, LR = self.dataset_loop(tr_loader, epoch, is_train=True, writer=writer, tb_tag='train')
            end_time = time.time()
            min_elapsed = (end_time - start_time) / 60
            print('Epoch {} -- train: loss={:.4f}, LKL={:.4f}, LR={:.4f}, time={:.0f}'\
                  .format(epoch, loss, LKL, LR, min_elapsed), file=stdout_f)
            stdout_f.flush()

            self.encoder_optimizer = self.lr_decay(self.encoder_optimizer, self.hp.min_lr, self.hp.lr_decay)
            self.decoder_optimizer = self.lr_decay(self.decoder_optimizer, self.hp.min_lr, self.hp.lr_decay)

            # validate
            self.encoder.eval()
            self.decoder.eval()
            loss, LKL, LR = self.dataset_loop(val_loader, epoch, is_train=False, writer=writer, tb_tag='valid')
            val_losses.append(loss)
            # if min(val_losses[-5:]) != loss:  # TODO: early stopping
            #     break
            print('Epoch {} -- valid: loss={:.4f}, LKL={:.4f}, LR={:.4f}'\
                  .format(epoch, loss, LKL, LR), file=stdout_f)
            stdout_f.flush()

            # Generate sequence to save image and show progress
            self.save_conditional_generation(cond_gen_data_loader, epoch, n_gens=1, output_imgs_path=output_imgs_path)

            # Save model
            # TODO: only save best model (model.pt)
            if loss < min_val_loss:
                min_val_loss = loss
                model_fn = 'e{}_loss{:.4f}.pt'.format(epoch, loss)  # val loss
                torch.save(self.state_dict(), os.path.join(run_path, model_fn))

        stdout_f.close()

    #
    # KL loss
    #
    def kullback_leibler_loss(self, sigma_hat, mu, KL_min, wKL, eta_step):
        """
        Calculate KL loss -- (eq. 10, 11)
        
        :param: sigma_hat: [bsz, Nz]
        :param: mu: [bsz, Nz]
        :param: KL_min: float
        :param: wKL: float
        :param: eta_step: float
        
        :returns: float Tensor
        """
        bsz, Nz = sigma_hat.size()

        LKL = -0.5 * torch.sum(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat)) \
              / float(bsz * Nz)
        KL_min = torch.Tensor([KL_min])
        if USE_CUDA:
            KL_min = KL_min.cuda()
        KL_min = KL_min.detach()

        LKL = wKL * eta_step * torch.max(LKL, KL_min)
        return LKL

    ##############################################################################
    # Conditional generation
    ##############################################################################
    def save_conditional_generation(self, data_loader, epoch, n_gens=1, output_imgs_path=None):
        """
        Generate sequence conditioned on output of encoder
        """
        n = 0
        for batch, lengths in data_loader:
            batch, lengths = self.preprocess_batch_from_data_loader(batch, lengths)
            max_len, bsz, _ = batch.size()

            # Encode
            z, _, _ = self.encoder(batch)  # z: [1, 1, 128]  # TODO: is z actually [1, 128]?

            # initialize state with start of sequence stroke-5 stroke
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            if USE_CUDA:
                sos = sos.cuda()

            # generate until end of sequence or maximum sequence length
            s = sos
            seq_x = []  # delta-x
            seq_y = []  # delta-y
            seq_pen = []  # pen-down
            hidden_cell = None
            for _ in range(max_len):
                input = torch.cat([s, z.unsqueeze(0)], 2)  # [1,1,133]

                # decode
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = \
                    self.decoder(input, z, output_all=False, hidden_cell=hidden_cell)
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
            output_fp = os.path.join(output_imgs_path, '{}-{}.jpg'.format(epoch, n))
            save_sequence_as_img(sequence, output_fp)

            n += 1
            if n == n_gens:
                break

    def sample_next_state(self, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
        """
        Return state using current mixture parameters etc. set from decoder call.
        Note that this state is different from the stroke-5 format.
        
        :param: pi: [len, bsz, M]
            # TODO!! This is currently hardcoded with save_conditional_generation
            # When we do pi.data[0,0,:] down below,
                The 0-th index 0: Currently, pi is being calculated with output_all=False, which means it's just ouputting the last pi.
                The 1-th index 0: first in batch
        # TODO: refactor so that the above isn't the case.
        
        :returns:
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
        if USE_CUDA:
            next_state = next_state.cuda()
        s = next_state.view(1, 1, -1)

        pen_down = q_idx == 1  # TODO: isn't this pen up?
        eos = q_idx == 2

        return s, dx, dy, pen_down, eos

    def sample_bivariate_normal(self, mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False):
        """
        Inputs are all floats

        Returns two floats
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_only', action='store_true')
    parser.add_argument('-c', '--category', type=str, default='cat')
    parser.add_argument('-n', '--model_name', type=str, default='sketchrnn')
    args = parser.parse_args()

    hp = HParams()

    if args.decoder_only:
        model = SketchRNNDecoderOnly(hp)
    else:
        model = SketchRNNVAE(hp)

    model.train_loop(args.model_name, args.category)
