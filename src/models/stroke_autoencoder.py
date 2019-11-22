# stroke_autoencoder.py

import os

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models.core.train_nn import TrainNN, RUNS_PATH
from src.models.core.transformer_utils import *
from src.models.base.instruction_models import ProgressionPairDataset, InstructionDecoderLSTM, \
    PAD_ID, OOV_ID, SOS_ID, EOS_ID
from src.models.base.stroke_models import StrokeEncoderTransformer, StrokeEncoderLSTM, StrokeEncoderCNN, \
    NpzStrokeDataset, NdjsonStrokeDataset, SketchRNNDecoderGMM

USE_CUDA = torch.cuda.is_available()

##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################

class HParams():
    def __init__(self):
        # Training
        self.batch_size = 256  # 100
        self.lr = 0.0001  # 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 1000

        # Model
        self.dim = 512
        self.n_enc_layers = 4
        self.n_dec_layers = 4
        self.enc_type = 'cnn'  # 'cnn', 'transformer, 'lstm'
        self.use_categories_enc = False
        self.use_categories_dec = False  # SketchRNN doesn't actually use categories
        self.dropout = 0.2
        self.M = 20

        # Other
        self.notes = ''

##############################################################################
#
# MODEL
#
##############################################################################

class StrokeAutoencoderModel(TrainNN):
    def __init__(self, hp, save_dir=None):
        super().__init__(hp, save_dir)

        self.tr_loader =  self.get_data_loader('train', self.hp.batch_size, shuffle=True)
        self.val_loader = self.get_data_loader('valid', self.hp.batch_size, shuffle=False)
        self.end_epoch_loader = self.val_loader

        # Model
        self.category_embedding = None
        if (self.hp.use_categories_enc) or (self.hp.use_categories_dec):
            self.category_embedding = nn.Embedding(35, self.hp.dim)
            self.models.append(self.category_embedding)

        if hp.enc_type == 'cnn':
            self.enc = StrokeEncoderCNN(n_feat_maps=hp.dim, input_dim=5, emb_dim=hp.dim, dropout=hp.dropout,
                                        use_categories=hp.use_categories_enc)
            # raise NotImplementedError('use_categories_enc=true not implemented for CNN encoder')
        elif hp.enc_type == 'transformer':
            self.enc = StrokeEncoderTransformer(5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout,
                                                use_categories=hp.use_categories_enc)
        elif hp.enc_type == 'lstm':
            self.enc = StrokeEncoderLSTM(5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout, batch_first=False,
                                         use_categories=hp.use_categories_enc)

        self.fc_z_to_hc =  nn.Linear(hp.dim, 2 * hp.dim)
        self.dec = SketchRNNDecoderGMM(5 + hp.dim, hp.dim, hp.M)

        self.models.extend([self.enc, self.fc_z_to_hc, self.dec])
        for model in self.models:
            model.cuda()
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def get_data_loader(self, dataset_split, batch_size, shuffle=True):
        """
        Uses the NpzStrokeDataset, all categories.
        
        Args:
            dataset_split: str
            batch_size: int
            categories: str
            shuffle: bool
        """
        ds = NdjsonStrokeDataset('all', dataset_split)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        return loader

    def preprocess_batch_from_data_loader(self, batch):
        """
        Transposes strokes, moves to cuda
        """
        strokes, stroke_lens, cats, cats_idx = batch
        strokes = strokes.transpose(0, 1).float()
        strokes = nn_utils.move_to_cuda(strokes)
        stroke_lens = stroke_lens.numpy().tolist()
        cats_idx = nn_utils.move_to_cuda(cats_idx)
        return strokes, stroke_lens, cats, cats_idx

    def encode_and_get_z_for_decoder(self, batch):
        """
        Returns:
            embedded: [bsz, dim] FloatTensor
            (hidden, cell): each is [n_layers, bsz, dim] FloatTensor
        """
        strokes, stroke_lens, cats, cats_idx = batch
        if self.hp.enc_type == 'cnn':
            z = self.enc(strokes, stroke_lens,
                                category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, dim]
        elif self.hp.enc_type == 'transformer':
            z = self.enc(strokes, stroke_lens,
                         category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, dim]
        elif self.hp.enc_type == 'lstm':
            _, (hidden, cell) = self.enc(strokes, stroke_lens,
                                         category_embedding=self.category_embedding, categories=cats_idx)
            # [bsz, max_stroke_len, dim]; h/c = [n_layers * n_directions (2), bsz, dim]
            # combine bidirections
            _, bsz, dim = hidden.size()
            hidden = hidden.view(self.enc.num_layers, 2, bsz, dim)
            hidden = hidden.mean(dim=1)  # [n_layers, bsz, dim]
            cell = cell.view(self.enc.num_layers, 2, bsz, dim)
            cell = cell.mean(dim=1)  # [n_layers, bsz, dim]

            # take last layer
            hidden, cell = hidden[-1], cell[-1]  # each [bsz, dim]; last layer
            # combine
            z = torch.stack([hidden, cell], dim=-1).mean(dim=-1)  # [bsz, dim]

        return z

    def one_forward_pass(self, batch):
        strokes, stroke_lens, cats, cats_idx = batch
        max_len, bsz, _ = strokes.size()

        # Encode
        z = self.encode_and_get_z_for_decoder(batch)  # each is [bsz, dim]

        # Create inputs to decoder
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)  # start of sequence  # [1, bsz, 5]
        sos = nn_utils.move_to_cuda(sos)
        inputs_init = torch.cat([sos, strokes], 0)  # add sos at the begining of the strokes; [max_len + 1, bsz, 5]
        z_stack = torch.stack([z] * (max_len + 1), dim=0)  # expand z to concat with inputs; [max_len + 1, bsz, z_dim]
        dec_inputs = torch.cat([inputs_init, z_stack], 2)  # each input is stroke + z; [max_len + 1, bsz, z_dim + 5]

        # init hidden and cell states is tanh(fc(z)) (Page 3)
        hidden, cell = torch.split(torch.tanh(self.fc_z_to_hc(z)), self.hp.dim, 1)
        hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())

        # Decode
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True,
                                                                     hidden_cell=hidden_cell)

        # Calculate losses
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)
        loss = self.dec.reconstruction_loss(mask,
                                            dx, dy, p,
                                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                            q)
        result = {'loss': loss, 'loss_R': loss}
        return result


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'stroke_autoencoder', run_name)
    model = StrokeAutoencoderModel(hp, save_dir)
    utils.save_run_data(save_dir, hp)
    model.train_loop()
