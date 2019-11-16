# sketch_rnn.py

"""
SketchRNN model as in "Neural Representation of Sketch Drawings"
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
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src import utils
from src.models.core import nn_utils
from src.models.base.stroke_models import StrokeDataset, SketchRNNDecoderGMM, SketchRNNVAEEncoder
from src.models.core.train_nn import TrainNN, RUNS_PATH


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
# Base SketchRNN model
#
##############################################################################

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
            cats: list of strs
            cats_idx: [bsz]
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


##############################################################################
#
# Decoder only ("Unconditional" model)
#
##############################################################################

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
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)
        loss = self.dec.reconstruction_loss(mask,
                                            dx, dy, p,
                                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                            q)
        if (pi != pi).any():
            import pdb; pdb.set_trace()
        result = {'loss': loss, 'loss_R': loss}

        return result

    def save_generation(self, gen_data_loader, epoch, n_gens=1, outputs_path=None):
        # TODO: generation not implemented yet (need to generalize the conditional generation of the VAE)
        pass


##############################################################################
#
# VAE Model ("Conditional" model)
#
##############################################################################

class SketchRNNVAEModel(SketchRNNModel):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """

    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        # Model
        self.enc = SketchRNNVAEEncoder(5, hp.enc_dim, hp.enc_num_layers, hp.z_dim, dropout=hp.dropout)
        self.fc_z_to_hc = nn.Linear(hp.z_dim, 2 * hp.dec_dim)  # 2: 1 for hidden, 1 for cell
        self.dec = SketchRNNDecoderGMM(hp.z_dim + 5, hp.dec_dim, hp.M)
        self.models.extend([self.enc, self.fc_z_to_hc, self.dec])
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
        hidden, cell = torch.split(torch.tanh(self.fc_z_to_hc(z)), self.hp.dec_dim, 1)
        hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        # TODO: if we want multiple layers, we need to replicate hidden and cell n_layers times

        # Decode
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True,
                                                                     hidden_cell=hidden_cell)

        # Calculate losses
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)
        loss_KL = self.kullback_leibler_loss(sigma_hat, mu, self.hp.KL_min, self.hp.wKL, self.eta_step)
        loss_R = self.dec.reconstruction_loss(mask,
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
            hidden, cell = torch.split(torch.tanh(self.fc_z_to_hc(z)), self.hp.dec_dim, 1)
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
