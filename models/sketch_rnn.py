# sketch_rnn.py

"""
SketchRNN model is VAE with GMM in decoder
"""

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

SKETCH_DATA = 'cat.npz'
OUTPUT_IMGS_PATH = 'outputs/'

use_cuda = torch.cuda.is_available()


##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        self.data_location = SKETCH_DATA

        # Training
        self.batch_size = 64
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.  # clip gradients greater than this

        # Model
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128  # dimension of z for VAE
        self.M = 20  # number of mixtures
        self.dropout = 0.9  #
        self.wKL = 0.5  # weight for Kl-loss (eq. 11)
        self.temperature = 0.4  # randomness (1=determinstic) in sampling (eq.8)
        self.max_seq_length = 200  # maximum number of strokes in a drawing

        # Annealing -- first focus more on reconstruction loss which is harder to optimize
        # I.e. once L_KL drops low enough,
        # eta_step = 1 - (1 - eta_min) * R
        # loss_train = loss_reconstruction + wKL * eta_step * max(loss_KL, KL_min)
        self.eta_min = 0.01  # initial annealing term(supp eq. 4)
        self.R = 0.99995  # annealing (supp eq. 4)
        self.KL_min = 0.2  #


hp = HParams()


##############################################################################
#
# DATASET
#
##############################################################################
class SketchDataset(Dataset):
    """
    Dataset to load sketches

    Notes
    -----
    Stroke-3 format: (delta-x, delta-y, binary for if pen is lifted)
    Stroke-5 format: consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
        one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
    """
    def __init__(self, data_path, transform=None):
        """

        """
        full_data = np.load(data_path, encoding='latin1')['train']  # e.g. cat.npz is in 3-stroke format
        self.data = self._preprocess_data(full_data)
        self.max_seq_len = max([len(seq[:, 0]) for seq in self.data])
        self.transform = transform

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
            if (seq_len > 10) and (seq_len <= hp.max_seq_length):
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
        n_batches = len(data) // hp.batch_size
        filtered = data[:n_batches * hp.batch_size]
        return filtered

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample, l = self._stroke3_to_stroke5(sample)

        if self.transform:
            sample = self.transform(sample)

        return (sample, l)

    def _stroke3_to_stroke5(self, seq):  # to_big_strokes() in sketch_rnn/utils.py
        """
        Convert from stroke-3 to stroke-5 format 
        """
        result = np.zeros((self.max_seq_len, 5), dtype=float)
        l = len(seq)
        assert l <= self.max_seq_len
        result[0:l, 0:2] = seq[:, 0:2]  # 1st and 2nd values are same
        result[0:l, 3] = seq[:, 2]  # stroke-5[3] = pen-up, same as stroke-3[2]
        result[0:l, 2] = 1 - result[0:l, 3]  # stroke-5[2] = pen-down, stroke-3[2] = pen-up (so inverse)
        result[l:, 4] = 1  # last "stroke" has stroke5[4] equal to 1, all other values 0 (see Figure 4); hence l
        return result, l


dataset = SketchDataset(hp.data_location)
Nmax = dataset.max_seq_len

##############################################################################
#
# UTILS
#
##############################################################################
def lr_decay(optimizer):
    """
    Decay learning rate by a factor of lr_decay
    """
    for param_group in optimizer.param_groups:
        if param_group['lr'] > hp.min_lr:
            param_group['lr'] *= hp.lr_decay
    return optimizer


def make_image(sequence, epoch, name='_output_'):
    """
    Plot drawing
    """
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()

    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    name = str(epoch) + name + '.jpg'
    output_path = os.path.join(OUTPUT_IMGS_PATH, name)
    pil_image.save(output_path, 'JPEG')
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

    def __init__(self):
        super(EncoderRNN, self).__init__()

        self.lstm = nn.LSTM(5, hp.enc_hidden_size,
                            dropout=hp.dropout, bidirectional=True)

        # Create mu and sigma by passing lstm's last output into fc layer (Eq. 2)
        self.fc_mu = nn.Linear(2 * hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2 * hp.enc_hidden_size, hp.Nz)

        # Active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        # Initialize hidden state and cell state with zeros on first forward pass
        if hidden_cell is None:
            hidden = torch.zeros(2, batch_size, hp.enc_hidden_size)
            cell = torch.zeros(2, batch_size, hp.enc_hidden_size)
            if use_cuda:
                hidden = hidden.cuda()
                cell = cell.cuda()
            hidden_cell = (hidden, cell)

        # Pass inputs, hidden, and cell into encoder's lstm
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTM
        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)

        # Get mu and sigma from hidden
        mu = self.fc_mu(hidden_cat)  # [bsz, Nz]
        sigma_hat = self.fc_sigma(hidden_cat)  # [bsz, Nz]

        # Get z for VAE using mu and sigma, N ~ N(0,1)
        # Turn sigma_hat vector into non-negative std parameter
        sigma = torch.exp(sigma_hat / 2.)
        z_size = mu.size()
        N = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        if use_cuda:
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
    Outputs:
        pi: weights for each mixture        [max_seq_len + 1, batch_size, num_mixtures]
        mu_x: mean x for each mixture;      [max_seq_len + 1, batch_size, num_mixtures]
        mu_y: mean y for each mixture;      [max_seq_len + 1, batch_size, num_mixtures]
        sigma_x: var x for each mixture;    [max_seq_len + 1, batch_size, num_mixtures]
        sigma_y: var y for each mixture;    [max_seq_len + 1, batch_size, num_mixtures]
        rho_xy:  covariance ''         ;    [max_seq_len + 1, batch_size, num_mixtures]
        q: models p (3 pen strokes in stroke-5) as categorical distribution (page 3);   [max_seq_len + 1, batch_size, 3]
        hidden: last hidden state;          [1, batch_size, dec_hidden_size]
        cell:   last cell state;            [1, batch_size, dec_hidden_size]
    """

    def __init__(self):
        super(DecoderRNN, self).__init__()

        # Initialize decoder states
        # Page 3: The initial hidden states h0, and optional cell states c0 (if applicable)
        # of the decoder RNN is the output of a single layer network [h0; c0] = tanh(W*z + b)
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)

        # Plus 5 for
        # TODO: is this the right equation / shoudl this comment be here
        # x_i = [S_{i-1}, z], [h_i; c_i] = forward(x_i, [h_{i-1}; c_{i-1}])     # Eq. 4
        self.lstm = nn.LSTM(hp.Nz + 5, hp.dec_hidden_size, dropout=hp.dropout)

        # Create mixture params and probs from hiddens
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)

    def forward(self, inputs, z, hidden_cell=None):
        # On first forward pass, initialize hidden state and cell state using z
        if hidden_cell is None:
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)

        # Pass hidden state at each step to fully connected layer
        # Fig. 2, Eq. 4
        # Dimensions
        #   outputs: [max_seq_len + 1, batch_size, dec_hidden_size]
        #   view: [(max_seq_len + 1) * batch_size, dec_hidden_size]
        #   y: [(max_seq_len + 1) * batch_size, 6 * num_mixtures + 3] (6 comes from 5 for params, 6th for weights; see page 3)
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))

        # Separate pen and mixture params
        params = torch.split(y, 6,
                             1)  # splits into tuple along 1st dim; tuple of num_mixture [(max_seq_len + 1) * batch_size, 6]'s, 1 [(max_seq_len + 1) * batch_size, 3]
        params_mixture = torch.stack(params[:-1])  # trajectories; [num_mixtures, (max_seq_len + 1) * batch_size, 6]
        params_pen = params[-1]  # pen up/down;  [(max_seq_len + 1) * batch_size, 3]

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
        if self.training:
            len_out = Nmax + 1
        else:
            len_out = 1


        # pi torch.Size([20, 8320])

        # TODO: don't think I actually need the squeeze's
        # if len_out == 1:
        #     import pdb; pdb.set_trace()  # squeeze may be related to generation with len_out = 1
        pi = F.softmax(pi.t().squeeze(), dim=-1).view(len_out, -1, hp.M)

        mu_x = mu_x.t().squeeze().contiguous().view(len_out, -1, hp.M)
        mu_y = mu_y.t().squeeze().contiguous().view(len_out, -1, hp.M)

        # Eq. 6
        sigma_x = torch.exp(sigma_x.t().squeeze()).view(len_out, -1, hp.M)
        sigma_y = torch.exp(sigma_y.t().squeeze()).view(len_out, -1, hp.M)
        rho_xy = torch.tanh(rho_xy.t().squeeze()).view(len_out, -1, hp.M)

        # Eq. 7
        q = F.softmax(params_pen, dim=-1).view(len_out, -1, 3)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell


##############################################################################
#
# MODEL
#
##############################################################################
class SketchModel():
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """

    def __init__(self):

        self.encoder = EncoderRNN()
        self.decoder = DecoderRNN()
        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

        # Optimization -- ADAM plus annealing (supp eq. 4)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

        # Bookkeeping
        self.run_datetime = datetime.now().strftime('%B%d_%H-%M-%S')

    ##############################################################################
    # Data
    ##############################################################################
    def make_target(self, batch, lengths):
        """
        Create target vector out of stroke-5 data and lengths. Namely, use lengths
        to create mask for each sequence. Detach "detaches" from computational graph
        so that backprop stops.

        Inputs
        ------
            batch:  [max_seq_len, batch_size, 5]
            lengths: list of ints 

        Params
        ------
            batch: Variable with Tensor that has data of size: [max_len, batch_size, 5]
            lengths: list of ints, number of strokes for each sequence in batch

        Outputs
        -------
            mask: [max_seq_len + 1, batch_size]
            dx: [max_seq_len + 1, batch_size, num_mixtures]
            dy: [max_seq_len + 1, batch_size, num_mixtures]
            p:  [max_seq_len + 1, batch_size, 3]

        """
        # Add eos
        eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).unsqueeze(0)  # ([1, batch_size, 5])
        if use_cuda:
            eos = eos.cuda()


        batch = torch.cat([batch, eos], 0)  # [max_len + 1, batch_size, 5]

        # Calculate mask for each sequence using lengths
        # Detach
        mask = torch.zeros(Nmax + 1, batch.size()[1])
        for indice, length in enumerate(lengths):
            mask[:length, indice] = 1
        if use_cuda:
            mask = mask.cuda()
        mask = mask.detach()
        dx = torch.stack([batch.data[:, :, 0]] * hp.M, 2).detach()
        dy = torch.stack([batch.data[:, :, 1]] * hp.M, 2).detach()
        p1 = batch.data[:, :, 2].detach()
        p2 = batch.data[:, :, 3].detach()
        p3 = batch.data[:, :, 4].detach()
        p = torch.stack([p1, p2, p3], 2)

        return mask, dx, dy, p

    def preprocess_batch_from_data_loader(self, batch, lengths):
        """
        Get in right dimension / format, Variablize, etc.

        Returns
        -------
        batch:  [max_seq_len, batch_size, 5]
        lengths: list of ints 
        """
        batch.transpose_(0, 1).float()  # Dataloader returns batch in 1st dimension, rest of code expects it to be 2nd
        if use_cuda:
            batch = batch.cuda()
        lengths = lengths.numpy().tolist()
        return batch, lengths

    ##############################################################################
    # Basic utils
    ##############################################################################
    # Utils
    def save(self, epoch):
        sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
                   'encoderRNN_sel_%3f_epoch_%d.pth' % (sel, epoch))
        torch.save(self.decoder.state_dict(), \
                   'decoderRNN_sel_%3f_epoch_%d.pth' % (sel, epoch))

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    ##############################################################################
    # Training
    ##############################################################################
    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()

        data_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)

        writer = SummaryWriter('runs/' + self.run_datetime)

        for i, (batch, lengths) in enumerate(data_loader):
            batch, lengths = self.preprocess_batch_from_data_loader(batch,
                                                                    lengths)  # batch = [max_seq_len, batch_size, 5]

            # Encode
            z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)  # each [batch_size, Nz]

            # Create inputs to decoder
            # First, create start of sequence
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).unsqueeze(0)
            if use_cuda:
                sos = sos.cuda()

            batch = batch.float()  # TODO: is this right? Also move to preprocess_batch()
            batch_init = torch.cat([sos, batch], 0)  # add sos at the begining of the batch; [max_seq_len + 1, batch_size, 5]
            z_stack = torch.stack([z] * (Nmax + 1),
                                  dim=0)  # expand z to be ready to concatenate with inputs; [max_seq_len + 1, batch_size, Nz]
            inputs = torch.cat([batch_init, z_stack],
                               2)  # each input is stroke + z; [max_seq_len + 1, batch_size, Nz + 5]

            # Decode
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)

            # Set up optimizers
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.eta_step = 1 - (1 - hp.eta_min) * hp.R  # update eta for LKL

            # Get targets
            mask, dx, dy, p = self.make_target(batch, lengths)

            # Calculate losses
            LKL = self.kullback_leibler_loss()
            LR = self.reconstruction_loss(mask, dx, dy, p)
            loss = LR + LKL

            # Gradient and optimization
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.grad_clip)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            # Logging
            if i % 10 == 0:
                writer.add_scalar('loss', loss.clone().cpu().data.numpy(),
                                  epoch * data_loader.__len__() + i)
                writer.add_scalar('reconstruction_loss', LR.clone().cpu().data.numpy(),
                                  epoch * data_loader.__len__() + i)
                writer.add_scalar('KL_loss', LKL.clone().cpu().data.numpy(),
                                  epoch * data_loader.__len__() + i)

        # Logging, decay learning rate potentially
        if epoch % 1 == 0:
            print('epoch=', epoch, 'loss=', loss.item(), 'LR=', LR.item(), 'LKL=', LKL.item())
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)

        # Generate sequence to save image and show progress
        if epoch % 1 == 0:
            # self.save(epoch)
            self.conditional_generation(epoch)

    ###### Reconstruction loss ######

    def reconstruction_loss(self, mask, dx, dy, p):
        """
        Eq. 9

        Params
        ------
        These are outputs from make_targets(batch, lengths)
            mask: [max_seq_len + 1, batch_size]
            dx: [max_seq_len + 1, batch_size, num_mixtures]
            dy: [max_seq_len + 1, batch_size, num_mixtures]
            p:  [max_seq_len + 1, batch_size, 3]

        + 1 because end of sequence stroke appended in make_targets()
        """

        # Loss w.r.t pen offset
        prob = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask * torch.log(1e-5 + torch.sum(self.pi * prob, 2))) \
             / float(Nmax * hp.batch_size)

        # Loss of pen parameters (cross entropy between ground truth pen params p
        # and predicted categorical distribution q)
        LP = -torch.sum(p * torch.log(self.q)) / float(Nmax * hp.batch_size)

        return LS + LP

    def bivariate_normal_pdf(self, dx, dy):
        """
        Get probability of dx, dy using mixture parameters. 

        Reference: Eq. of https://arxiv.org/pdf/1308.0850.pdf (Graves' Generating Sequences with 
        Recurrent Neural Networks)
        """
        # Eq. 25
        # Reminder: mu's here are calculated for mixture model on the stroke data, which
        # models delta-x's and delta-y's. So z_x just comparing actual ground truth delta (dx)
        # to the prediction from the mixture model (mu_x). Then normalizing etc.
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy

        # Eq. 24
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2)
        exp = torch.exp(-z / (2 * (1 - self.rho_xy ** 2)))

        return exp / norm

    ###### KL loss ######

    def kullback_leibler_loss(self):
        """
        Calcualte KL loss -- (eq. 10, 11)
        """
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma)) \
              / float(hp.Nz * hp.batch_size)
        KL_min = torch.Tensor([hp.KL_min])
        if use_cuda:
            KL_min = KL_min.cuda()
        KL_min = KL_min.detach()

        return hp.wKL * self.eta_step * torch.max(LKL, KL_min)

    ##############################################################################
    # Conditional generation
    ##############################################################################
    def conditional_generation(self, epoch):
        """
        Generate sequence conditioned on output of encoder
        """

        self.encoder.train(False)
        self.decoder.train(False)

        data_loader = DataLoader(dataset, batch_size=1)

        for i, (batch, lengths) in enumerate(data_loader):
            batch, lengths = self.preprocess_batch_from_data_loader(batch, lengths)

            # TODO: save original image here (this is conditional reconstruction)

            # Encode
            # should remove dropouts
            z, _, _ = self.encoder(batch, 1)  # z: [1, 1, 128]  # TODO: is z actually [1, 128]?

            # # # Unconditional reconstruction (note that this is not the same as training with decoder only though
            # z = torch.zeros(1, hp.Nz)
            # # z = torch.zeros(1, 1, 128)
            # # # z = torch.rand(1, 1, 128)
            # if use_cuda:
            #     z = z.cuda()

            # Initialize state with start of sequence stroke-5 stroke
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            if use_cuda:
                sos = sos.cuda()

            s = sos

            # Generate until end of sequence or maximum sequence length
            seq_x = []  # delta-x
            seq_y = []  # delta-y
            seq_pen = []  # pen-down
            hidden_cell = None
            for i in range(Nmax):
                input = torch.cat([s, z.unsqueeze(0)], 2)  # [1,1,133]

                # Decode
                self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(input, z, hidden_cell)
                hidden_cell = (hidden, cell)

                # Sample next state
                s, dx, dy, pen_down, eos = self.sample_next_state()  # not quite stroke-5
                seq_x.append(dx)
                seq_y.append(dy)
                seq_pen.append(pen_down)

                # Done drawing
                if eos:
                    break

            # Get in format to draw image
            # Cumulative sum because seq_x and seq_y are deltas, so get x (or y) at each stroke
            sample_x = np.cumsum(seq_x, 0)
            sample_y = np.cumsum(seq_y, 0)
            sample_pen = np.array(seq_pen)
            sequence = np.stack([sample_x, sample_y, sample_pen]).T
            make_image(sequence, epoch)

            # Just generate one sample (currently only calling this at end of train epoch to save image)
            break

    def sample_next_state(self):
        """
        Return state using current mixture parameters etc. set from decoder call.
        Note that this state is different from the stroke-5 format.
        State is [x, y, 

        NOTE: sampling is done on the first sequence in batch (hence the 0 indexing)
        """

        def adjust_temp(pi_pdf):
            """Not super sure why this instead of just dividing by temperauture as in eq. 8, but
            magenta sketch_run/model.py does it this way(adjust_temp())"""
            pi_pdf = np.log(pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # Get mixture index
        # TODO: make this function take in pi?
        # pi is weights for mixtures
        # This is only taking the pi for the first (seq_len, batch_size, num_mixtures)
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)  # choose Gaussian weighted by pi

        # Get pen state
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # print(q_idx)

        # Get mixture params
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]

        # Get next x andy by using mixture params and sampling from bivariate normal
        x, y = self.sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)

        # Create next_state vector
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1

        if use_cuda:
            next_state = next_state.cuda()

        return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2

    def sample_bivariate_normal(self, mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False):
        """
        Inputs are all floats

        Returns two floats
        """
        if greedy:
            return mu_x, mu_y

        mean = [mu_x, mu_y]

        # Randomness controlled by temperature (eq. 8)
        sigma_x *= np.sqrt(hp.temperature)
        sigma_y *= np.sqrt(hp.temperature)

        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
               [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]

        # TODO: is this right
        mean = [v.item() for v in mean]
        for i, row in enumerate(cov):
            cov[i] = [v.item() for v in row]
        x = np.random.multivariate_normal(mean, cov, 1)

        return x[0][0], x[0][1]


if __name__ == "__main__":
    model = SketchModel()
    for epoch in range(100):
        model.train(epoch)
