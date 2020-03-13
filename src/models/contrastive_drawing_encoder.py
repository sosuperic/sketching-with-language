"""
Usage:
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python src/models/contrastive_drawing_encoder.py
"""

from datetime import datetime
import os
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import RUNS_PATH
from src.models.base.instruction_models import NdjsonStrokeDataset
from src.models.base.stroke_models import StrokeEncoderLSTM
from src.models.core.train_nn import TrainNN
from src.models.core import experiments, nn_utils

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
        self.max_epochs = 1000

        # Dataset (and model)
        self.max_per_category = 7000
        self.drawing_type = 'stroke'  # 'stroke' or 'image'
        self.cnn_type = 'wideresnet'  #'wideresnet,se,cbam' (when drawing_type == 'image')
        self.use_prestrokes = False  # for 'stroke'
        self.images = 'pre,start_to_annotated,full'  # for image; annotated,pre,post,start_to_annotated,full

        # Model
        self.dim = 256
        self.n_enc_layers = 4
        self.model_type = 'lstm'  # 'lstm', 'transformer_lstm', 'cnn_lstm'
        self.use_layer_norm = False   # currently only for lstm
        self.use_categories_enc = False
        self.dropout = 0.2
        self.loss_tau = 0.1

        # Other
        self.notes = ''


class ContrastiveDrawingEncoderModel(TrainNN):
    def __init__(self, hp, save_dir=None):
        super().__init__(hp, save_dir)
        self.tr_loader = self.get_data_loader('train', shuffle=True)
        self.val_loader = self.get_data_loader('valid', shuffle=False)
        self.end_epoch_loader = self.val_loader

        #
        # Model
        #
        self.category_embedding = None
        if self.hp.use_categories_enc:
            self.category_embedding = nn.Embedding(35, hp.dim)
            self.models.append(self.category_embedding)

        # Encoder decoder
        if hp.model_type.endswith('lstm'):
            if hp.drawing_type == 'image':
                self.n_channels = len(hp.images.split(','))
                self.enc = StrokeAsImageEncoderCNN(hp.cnn_type, self.n_channels, hp.dim)
            else:  # drawing_type is stroke

                # encoders may be different
                if hp.model_type == 'cnn_lstm':
                    self.enc = StrokeEncoderCNN(n_feat_maps=hp.dim, input_dim=5, emb_dim=hp.dim, dropout=hp.dropout,
                                                use_categories=hp.use_categories_enc)
                    # raise NotImplementedError('use_categories_enc=true not implemented for CNN encoder')
                elif hp.model_type == 'transformer_lstm':
                    self.enc = StrokeEncoderTransformer(
                        5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout,
                        use_categories=hp.use_categories_enc,
                    )
                elif hp.model_type == 'lstm':
                    self.enc = StrokeEncoderLSTM(
                        5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout, batch_first=False,
                        use_categories=hp.use_categories_enc, use_layer_norm=hp.use_layer_norm
                    )
            self.models.append(self.enc)

        for model in self.models:
            model.cuda()

        # Optimizers
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    #
    # Data
    #
    def get_data_loader(self, dataset_split, shuffle=False):
        """
        Args:
            dataset_split (str): 'train', 'valid', 'test'
            shuffle (bool)
        """
        if self.hp.drawing_type == 'stroke':
            ds = NdjsonStrokeDataset('all', dataset_split,
                                     max_per_category=self.hp.max_per_category,
                                     must_have_instruction_tree=False)
            loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=shuffle)
        elif self.hp.drawing_type == 'image':
            # TODO: need to save images for ndjson
            raise NotImplementedError
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

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        Args: batch: tuple from DataLoader
            strokes: [max_stroke_len, bsz, 5] FloatTensor
            stroke_lens: list of ints
            texts: list of strs
            text_lens: list of ints
            text_indices_w_sos_eos: [max_text_len + 2, bsz] LongTensor (+2 for sos and eos)
            cats: list of strs (categories)
            cats_idx: list of ints

        :return: dict: 'loss': float Tensor must exist
        """
        if self.hp.drawing_type == 'image':
            raise NotImplementedError
        elif self.hp.drawing_type == 'stroke':
            if self.hp.model_type == 'cnn_lstm':
                raise NotImplementedError
            elif self.hp.model_type == 'transformer_lstm':
                raise NotImplementedError
            elif self.hp.model_type == 'lstm':
                return self.one_forward_pass_lstm(batch)
            elif self.hp.model_type == 'transformer':
                raise NotImplementedError

    def data_augment_strokes(self, strokes, stroke_lens, methods='jitter,slice'):
        """
        Used to generative positive and negative examples.

        Args:
            strokes ([max_len, bsz, 5])
            stroke_lens
            methods (str): comma separated list
                jitter: multiply x y positions by [0.9,1.1]
                slice:
        """
        augmented = strokes.clone()
        aug_lens = None
        max_len, bsz, _ = strokes.size()

        methods = methods.split(',')
        if 'jitter' in methods:
            jitter = torch.zeros(max_len, bsz).uniform_(0.8,1.2)
            augmented[:,:,:2] *= jitter.to(strokes.get_device()).unsqueeze(2)
        if 'slice' in methods:
            aug_lens = []
            for i in range(bsz):
                penups = np.where(strokes[:,i,:].cpu().numpy()[:,3] == 1)[0].tolist()
                if len(penups) > 2:
                    end_slice_idx = random.choice(penups[1:-1])     # slice until end_slice_idx
                    augmented[end_slice_idx+1:,i,:] = 0.0     # zero eveything out
                    augmented[end_slice_idx+1:,i,4] = 1.0     # except for the drawing done, which should be 1's
                    aug_lens.append(end_slice_idx + 2)
                else:
                    aug_lens.append(stroke_lens[i])

        return augmented, aug_lens


    def one_forward_pass_lstm(self, batch):
        strokes, stroke_lens, cats, cats_idx = batch

        # Encode strokes
        _, (hidden, cell) = self.enc(strokes, stroke_lens,
                                    category_embedding=self.category_embedding, categories=cats_idx)
        # [bsz, max_stroke_len, dim]; h/c = [n_layers, bsz, dim]
        strokes_enc = hidden[-1,:,:]  # [bsz, dim]

        # Get augmented samples
        aug_strokes, aug_stroke_lens = self.data_augment_strokes(strokes, stroke_lens)  # [bsz, ]
        _, (aug_hidden, aug_cell) = self.enc(aug_strokes, aug_stroke_lens,
                                     category_embedding=self.category_embedding, categories=cats_idx)
        aug_strokes_enc = aug_hidden[-1,:,:]  # [bsz, dim]

        # Contrastive loss
        loss = self.contrastive_loss(strokes_enc, aug_strokes_enc)
        result = {'loss': loss}

        return result

    def contrastive_loss(self, strokes_enc, aug_strokes_enc):
        """
        Contrastive loss based on https://arxiv.org/pdf/2002.05709.pdf (eq 1)

        Table 5 looks at effect of l2 normalization and different temperature

        Args:
            strokes_enc ([bsz, dim]):
            aug_strokes_enc ([bsz, dim]):
        """
        bsz = strokes_enc.size(0)

        # Apply l2 normalization (see Table 5)
        strokes_norm = strokes_enc.norm(p=2, dim=1, keepdim=True)
        aug_norm = aug_strokes_enc.norm(p=2, dim=1, keepdim=True)
        strokes_enc = strokes_enc / strokes_norm
        aug_strokes_enc = aug_strokes_enc / aug_norm

        # Compute pairwise simlarities
        all_enc = torch.cat([strokes_enc, aug_strokes_enc], dim=0)  # [2 * bsz, dim]
        sims = nn_utils.cosine_sim(all_enc)  # [2 * bsz, 2 * bsz]

        # There is ONE positive pair of examples for each item in bsz
        # All the other items are negative examples
        # Except for don't compare against itself (hence the -1)
        matrix = torch.zeros(bsz, 2 * bsz - 1).to(sims.get_device())
        for i in range(bsz):
            cur_sims = sims[i]  # [2 * bsz]
            cur_sims = torch.cat([cur_sims[:i], cur_sims[i+1:]])  # [2 * bsz - 1]
            matrix[i,:] = cur_sims

        matrix = -torch.log(F.softmax(matrix / self.hp.loss_tau, dim=1))  # [bsz, 2 * bsz -1]

        # Actual loss looks at -log softmax value of positive examples
        loss = 0.0
        for i in range(bsz):
            loss += matrix[i, i + bsz-1]  # -1 because the sim between itself was ignored
        loss /= bsz

        return loss

    # End of epoch hook
    def end_of_epoch_hook(self, data_loader, epoch, outputs_path=None, writer=None):
        """
        Args:
            data_loader: DataLoader
            epoch: int
            outputs_path: str
            writer: Tensorboard Writer
        """
        pass


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'contrastive_drawing_encoder', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
    model = ContrastiveDrawingEncoderModel(hp, save_dir)
    experiments.save_run_data(save_dir, hp)
    model.train_loop()
