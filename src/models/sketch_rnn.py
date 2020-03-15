# sketch_rnn.py

"""
SketchRNN model as in "Neural Representation of Sketch Drawings"

Usage:
    PYTHONPATH=. python src/models/sketch_rnn.py --model_type vae
    PYTHONPATH=. python src/models/sketch_rnn.py --model_type decodergmm

PYTHONPATH=. python src/models/sketch_rnn.py --inference --temperature 0.1 \
--load_model_path runs/sketchrnn/Mar13_2020/drawings/categories_pig-dataset_ndjson-dec_dim_2048-enc_dim_512-enc_num_layers_1-lr_0.0001-max_per_category_70000-model_type_decodergmm-use_categories_dec_True/
"""

from datetime import datetime
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import RUNS_PATH
from src import utils
from src.data_manager.quickdraw import save_strokes_as_img, save_multiple_strokes_as_img
from src.models.base.instruction_models import (ProgressionPairDataset)
from src.models.base.stroke_models import (NdjsonStrokeDataset,
                                           NpzStrokeDataset,
                                           SketchRNNDecoderGMM,
                                           SketchRNNDecoderLSTM,
                                           SketchRNNVAEEncoder)
from src.models.core import experiments, nn_utils
from src.models.core.train_nn import TrainNN

USE_CUDA = torch.cuda.is_available()


##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        # Data
        self.dataset = 'ndjson'  # 'progressionpair' or 'ndjson'
        self.max_len = 200
        self.max_per_category = 2500
        self.categories = 'all'  # used with dataset='ndjson', comma separated categories or 'all'

        # Training
        self.batch_size = 64  # 100
        self.lr = 0.001  # 0.0001 when enc_dim=512, dec_dim=2048
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 100

        # Model
        self.model_type = 'decodergmm'  # 'vae', 'decodergmm', 'decoderlstm'
        self.use_layer_norm = False
        self.rec_dropout = 0.1  # only with use_layer_norm=True
        self.use_categories_dec = False
        self.categories_dim = 256
        self.enc_dim = 512  # 512
        self.dec_dim = 512  # 2048
        self.enc_num_layers = 1  # 2
        self.z_dim = 128  # dimension of z for VAE
        self.M = 20  # number of mixtures
        self.dropout = 0.1
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
# Base SketchRNN model
#
##############################################################################

class SketchRNNModel(TrainNN):
    def __init__(self, hp, save_dir, skip_data=False):
        super().__init__(hp, save_dir)

        self.eta_step = hp.eta_min
        if not skip_data:
            self.tr_loader = self.get_data_loader('train', hp.batch_size, hp.categories, hp.max_len, hp.max_per_category, True)
            self.val_loader = self.get_data_loader('valid', hp.batch_size, hp.categories, hp.max_len, hp.max_per_category, False)
            self.end_epoch_loader = self.get_data_loader('train', 1, hp.categories, hp.max_len, hp.max_per_category, True)

    #
    # Data
    #
    def get_data_loader(self, dataset_split, batch_size, categories, max_len, max_per_category, shuffle):
        """
        Args:
            dataset_split (str): 'train', 'valid', 'test'
            batch_size (int)
            categories (str)
            shuffle (bool)
        """
        if self.hp.dataset == 'ndjson':
            ds = NdjsonStrokeDataset(categories, dataset_split, max_len=max_len, max_per_category=max_per_category,
                                     must_have_instruction_tree=True)  # must_have...=True for fair comparison with planning models
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        elif self.hp.dataset == 'progressionpair':
            # We are using the ProgressionPair dataset, which has segments annotated.
            # In this case, we aren't using the segments or the instructions. We are using the full strokes.
            ds = ProgressionPairDataset(dataset_split, use_full_drawings=True, max_length=max_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=ProgressionPairDataset.collate_fn_strokes_categories_only)
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

    def end_of_epoch_hook(self, data_loader, epoch, outputs_path=None, writer=None):
        if self.hp.model_type == 'decodergmm':
            self.generate_and_save(data_loader, epoch, 25, outputs_path=outputs_path)

    ##############################################################################
    # Generate
    ##############################################################################
    def generate_and_save(self, data_loader, epoch, n_gens, outputs_path=None):
        """
        Generate sequence
        """
        n = 0
        gen_strokes = []
        gt_strokes = []
        for i, batch in enumerate(data_loader):
            batch = self.preprocess_batch_from_data_loader(batch)
            strokes, stroke_lens, cats, cats_idx = batch

            max_len, bsz, _ = strokes.size()

            # Encode
            if self.hp.model_type == 'vae':
                z, _, _ = self.enc(strokes)  # z: [bsz, 128]
                # init hidden and cell states is tanh(fc(z)) (Page 3)
                hidden, cell = torch.split(torch.tanh(self.fc_z_to_hc(z)), self.hp.dec_dim, dim=1)
                hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
            elif 'decoder' in self.hp.model_type:
                hidden_cell = (nn_utils.move_to_cuda(torch.zeros(1, bsz, self.hp.dec_dim)),
                               nn_utils.move_to_cuda(torch.zeros(1, bsz, self.hp.dec_dim)))

            # initialize state with start of sequence stroke-5 stroke
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)  # [1 (len), bsz, 5 (stroke-5)]
            sos = nn_utils.move_to_cuda(sos)

            # generate until end of sequence or maximum sequence length
            s = sos
            seq_x = []  # delta-x
            seq_y = []  # delta-y
            seq_pen = []  # pen-down
            for _ in range(max_len):
                if self.hp.model_type == 'vae':  # input is last state, z, and hidden_cell
                    input = torch.cat([s, z.unsqueeze(0)], dim=2)  # [1 (len), 1 (bsz), input_dim (5) + z_dim (128)]

                elif self.hp.model_type == 'decodergmm':  # input is last state and hidden_cell
                    input = s   # [1, bsz (1), 5]
                    if self.hp.use_categories_dec:
                        cat_embs = self.category_embedding(cats_idx)  # [bsz (1), cat_dim]
                        input = torch.cat([input, cat_embs.unsqueeze(0)], dim=2)  # [1, 1, cat_dim + 5]
                    outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = \
                        self.dec(input, stroke_lens=stroke_lens, output_all=False, hidden_cell=hidden_cell)
                    hidden_cell = (hidden, cell)  # for next timie step
                    # sample next state
                    s, dx, dy, pen_up, eos = self.sample_next_state(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)

                elif self.hp.model_type == 'decoderlstm':  # input is last state and hidden_cell
                    input = s
                    xy, q, hidden, cell = self.dec(input, stroke_lens=stroke_lens, output_all=False, hidden_cell=hidden_cell)
                    hidden_cell = (hidden, cell)
                    dx, dy = xy[-1,0,0].item(), xy[-1,0,1].item()  # last timestep, first batch item, x / y
                    pen_up = q[-1,0,:].max(dim=0)[1].item() == 1  # max index is the 2nd one (penup)
                    eos = q[-1,0,:].max(dim=0)[1].item() == 2  # max index is the 3rd one (eos)

                seq_x.append(dx)
                seq_y.append(dy)
                seq_pen.append(pen_up)

                if eos:  # done drawing
                    break

            # get in format to draw image
            # Cumulative sum because seq_x and seq_y are deltas, so get x (or y) at each stroke
            sample_x = np.cumsum(seq_x, 0)
            sample_y = np.cumsum(seq_y, 0)
            sample_pen = np.array(seq_pen)
            sequence = np.stack([sample_x, sample_y, sample_pen]).T
            output_fp = os.path.join(outputs_path, f'e{epoch}-gen{n}.jpg')
            save_strokes_as_img(sequence, output_fp)

            # Save original as well
            output_fp = os.path.join(outputs_path, f'e{epoch}-gt{n}.jpg')
            strokes_x = strokes[:, 0,
                        0]  # first 0 for x because sample_next_state etc. only using 0-th batch item; 2nd 0 for dx
            strokes_y = strokes[:, 0, 1]  # 1 for dy
            strokes_x = np.cumsum(strokes_x.cpu().numpy())
            strokes_y = np.cumsum(strokes_y.cpu().numpy())
            strokes_pen = strokes[:, 0, 3].cpu().numpy()
            strokes_out = np.stack([strokes_x, strokes_y, strokes_pen]).T
            save_strokes_as_img(strokes_out, output_fp)

            gen_strokes.append(sequence)
            gt_strokes.append(strokes_out)

            n += 1
            if n == n_gens:
                break

        rowcol_size = 5
        chunk_size = rowcol_size ** 2
        for i in range(0, chunk_size, len(gen_strokes)):
            output_fp = os.path.join(outputs_path, f'e{epoch}_gen{i}-{i+chunk_size}.jpg')
            save_multiple_strokes_as_img(gen_strokes[i:i+chunk_size], output_fp)

            output_fp = os.path.join(outputs_path, f'e{epoch}_gt{i}-{i+chunk_size}.jpg')
            save_multiple_strokes_as_img(gt_strokes[i:i+chunk_size], output_fp)



    def sample_next_state(self, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
        """
        Return state using current mixture parameters etc. set from decoder call.
        Note that this state is different from the stroke-5 format.

        NOTE: currently only operates on first item in batch (hence the BATCH_ITEM)

        Args:
            pi: [len, bsz, M]
            mu_x: [len, bsz, M]
            mu_y: [len, bsz, M]
            sigma_x: [len, bsz, M]
            sigma_y: [len, bsz, M]
            rho_xy: [len, bsz, M]
            q: [len, bsz, 3]

            When used during generation, len should be 1 (decoding step by step)

        Returns:
            s: [1, (bsz), 5]
            dx: [1]
            dy: [1]
            pen_up: bool
            eos: bool
        """

        def adjust_temp(pi_pdf):
            """Not super sure why this instead of just dividing by temperauture as in eq. 8, but
            magenta sketch_run/model.py does it this way(adjust_temp())"""
            pi_pdf = np.log(pi_pdf) / self.hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        _, bsz, M = pi.size()

        # TODO: currently, this method (and sample_bivariate_normal) only doesn't work produce samples
        # for every item in batch. It only does it for BATCH_ITEM-th point.
        BATCH_ITEM = 0  # index in batch

        # Get mixture index
        pi = pi.data[-1, BATCH_ITEM, :].cpu().numpy()  # [M]
        pi = adjust_temp(pi)

        pi_idx = np.random.choice(M, p=pi)  # choose Gaussian weighted by pi

        # Get mixture params
        mu_x = mu_x.data[-1, BATCH_ITEM, pi_idx]  # [M]
        mu_y = mu_y.data[-1, BATCH_ITEM, pi_idx]  # [M]
        sigma_x = sigma_x.data[-1, BATCH_ITEM, pi_idx]  # [M]
        sigma_y = sigma_y.data[-1, BATCH_ITEM, pi_idx]  # [M]
        rho_xy = rho_xy.data[-1, BATCH_ITEM, pi_idx]  # [M]

        # Get next x andy by using mixture params and sampling from bivariate normal
        dx, dy = self.sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)

        # Get pen state
        q = q.data[-1, BATCH_ITEM, :].cpu().numpy()  # [3]
        # q = adjust_temp(q)  # TODO: they don't adjust the temp for q in the magenta repo...
        q_idx = np.random.choice(3, p=q)

        # Create next_state vector
        next_state = torch.zeros(5)
        next_state[0] = dx
        next_state[1] = dy
        next_state[q_idx + 2] = 1
        next_state = nn_utils.move_to_cuda(next_state)
        s = next_state.view(1, 1, -1)

        pen_up = q_idx == 1
        eos = q_idx == 2

        return s, dx, dy, pen_up, eos

    def sample_bivariate_normal(self, mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False):
        """
        Args:
            mu_x: [M]
            mu_y: [M]
            sigma_x: [M]
            sigma_y: [M]
            rho_xy: [M]
            greedy: bool

        Return:
            Tuple of floats
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


    ##############################################################################
    #
    # Testing / Inference
    #
    ##############################################################################
    def save_imgs_inference_time(self, model_dir):
        n_gens = 25
        loader = self.get_data_loader('train', 1, self.hp.categories, self.hp.max_len, n_gens, True)
        outputs_path = os.path.join(opt.load_model_path, 'inference', str(self.hp.temperature))
        os.makedirs(outputs_path, exist_ok=True)
        print('Saving images to: ', outputs_path)
        torch.backends.cudnn.benchmark = True # Optimizes cudnn
        with torch.no_grad():
            self.generate_and_save(loader, 'dummy', n_gens, outputs_path=outputs_path)


##############################################################################
#
# Decoder only ("Unconditional" model)
#
##############################################################################

class SketchRNNDecoderLSTMOnlyModel(SketchRNNModel):
    def __init__(self, hp, save_dir, skip_data=False):
        super().__init__(hp, save_dir, skip_data=skip_data)

        # Model
        self.dec = SketchRNNDecoderLSTM(5, hp.dec_dim, dropout=hp.dropout)
        self.models.append(self.dec)
        if USE_CUDA:
            for model in self.models:
                model.cuda()
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
        dec_inputs = torch.cat([sos, strokes], dim=0)  # add sos at the begining of the strokes; [max_len + 1, bsz, 5]

        # Decode
        xy, q, hidden, cell = self.dec(dec_inputs, stroke_lens=stroke_lens)

        # Calculate losses
        mask, dxdy, p = self.dec.make_target(strokes, stroke_lens)

        loss = self.dec.reconstruction_loss(mask, dxdy, p,
                                            xy, q)
        result = {'loss': loss, 'loss_R': loss}

        return result


class SketchRNNDecoderGMMOnlyModel(SketchRNNModel):
    def __init__(self, hp, save_dir, skip_data=False):
        super().__init__(hp, save_dir, skip_data=skip_data)

        # Model
        self.category_embedding = None
        if hp.use_categories_dec:
            self.category_embedding = nn.Embedding(35, self.hp.categories_dim)
            self.models.append(self.category_embedding)
        inp_dim = (5 + hp.categories_dim) if self.category_embedding else 5
        self.dec = SketchRNNDecoderGMM(inp_dim, hp.dec_dim, hp.M, dropout=self.hp.dropout,
            use_layer_norm=self.hp.use_layer_norm, rec_dropout=self.hp.rec_dropout)
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

        if self.hp.use_categories_dec:
            cat_embs = self.category_embedding(cats_idx)  # [bsz, cat_dim]
            cat_embs = cat_embs.repeat(dec_inputs.size(0), 1, 1)  # [max_len + 1, bsz, cat_dim]
            dec_inputs = torch.cat([dec_inputs, cat_embs], dim=2)  # [max_len+1, bsz, 5 + cat_dim]

        # Decode
        outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)

        # Calculate losses
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)

        loss = self.dec.reconstruction_loss(mask,
                                            dx, dy, p,
                                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                            q)
        result = {'loss': loss, 'loss_R': loss}

        if ((loss != loss).any() or (loss == float('inf')).any() or (loss == float('-inf')).any()):
            raise Exception('Nan in SketchRNnDecoderGMMOnly forward pass')

        return result


##############################################################################
#
# VAE Model ("Conditional" model)
#
##############################################################################

class SketchRNNVAEModel(SketchRNNModel):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """

    def __init__(self, hp, save_dir, skip_data=False):
        super().__init__(hp, save_dir, skip_data=skip_data)

        # Model
        self.enc = SketchRNNVAEEncoder(5, hp.enc_dim, hp.enc_num_layers, hp.z_dim, dropout=hp.dropout)
        self.fc_z_to_hc = nn.Linear(hp.z_dim, 2 * hp.dec_dim)  # 2: 1 for hidden, 1 for cell
        self.dec = SketchRNNDecoderGMM(hp.z_dim + 5, hp.dec_dim, hp.M, dropout=hp.dropout)
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
        _, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True,
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


if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load_model_path', help='path to directory containing model to load for inference')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'sketchrnn', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)

    # If inference, load hparams
    if opt.inference:
        temp = hp.temperature  # store this because we will vary at inference
        orig_hp = utils.load_file(os.path.join(opt.load_model_path, 'hp.json'))  # dict
        for k, v in orig_hp.items():
            if k != 'temperature':
                setattr(hp, k, v)
    else:
        experiments.save_run_data(save_dir, hp)


    model = None
    if hp.model_type == 'vae':
        model = SketchRNNVAEModel(hp, save_dir, skip_data=opt.inference)
    elif hp.model_type == 'decodergmm':
        model = SketchRNNDecoderGMMOnlyModel(hp, save_dir, skip_data=opt.inference)
    elif hp.model_type == 'decoderlstm':
        model = SketchRNNDecoderLSTMOnlyModel(hp, save_dir, skip_data=opt.inference)
    model = nn_utils.AccessibleDataParallel(model)

    if opt.inference:
        model.load_model(opt.load_model_path)
        setattr(model.hp, 'temperature', temp)  # this may vary at inference time
        model.save_imgs_inference_time(opt.load_model_path)
    else:
        model.train_loop()
