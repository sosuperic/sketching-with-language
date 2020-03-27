"""
Usage:
PYTHONPATH=. python src/models/instruction_to_vaez.py
"""


from collections import defaultdict
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import RUNS_PATH
from src import utils
# from src.data_manager.quickdraw import build_category_index_nodata
from src.models.base.instruction_models import InstructionVAEzDataset, InstructionEncoderTransformer
from src.models.core import experiments, nn_utils
from src.models.core.train_nn import TrainNN

USE_CUDA = torch.cuda.is_available()


class HParams():
    def __init__(self):
        super().__init__()

        # Data and training
        self.dataset = 'ndjson'   # 'progressionpair' or 'ndjson'
        self.categories = 'pig'
        self.max_per_category = 70000
        self.prob_threshold = 0.0  # prune trees
        self.batch_size = 64

        self.lr = 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.0
        self.max_epochs = 100

        # Model
        self.enc_dim = 256
        self.enc_num_layers = 4
        self.dropout = 0.1

        # Keep this fixed
        self.categories_dim = 256
        self.z_dim = 128

class InstructionToVAEzModel(TrainNN):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)
        self.tr_loader = self.get_data_loader('train', True)
        self.val_loader = self.get_data_loader('valid', False)
        self.end_epoch_loader = None

        # Model
        self.text_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.enc_dim)
        self.category_embedding = nn.Embedding(35, 	hp.categories_dim)
        self.enc = InstructionEncoderTransformer(hp.enc_dim, hp.enc_num_layers, hp.dropout,
            use_categories=True, categories_dim=hp.categories_dim)
        self.fc_enc_to_zdim = nn.Linear(hp.enc_dim, hp.z_dim)

        for model in [self.text_embedding, self.category_embedding, self.enc, self.fc_enc_to_zdim]:
            model.cuda()

        # Optimizer
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def get_data_loader(self, dataset_split, shuffle):
        ds = InstructionVAEzDataset(dataset_split=dataset_split,
                                    categories=self.hp.categories, max_per_category=self.hp.max_per_category)
        loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=shuffle,
                            collate_fn=InstructionVAEzDataset.collate_fn)
        return loader

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        Args:
            batch: tuple from loader

        Returns:
            dict where 'loss': float Tensor must exist
        """
        texts, text_lens, text_indices, cats, cats_idx, vae_zs = batch
        instruction_embs = self.enc(text_indices, text_lens, self.text_embedding,
                                    category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, enc_dim]
        instruction_embs = self.fc_enc_to_zdim(instruction_embs)  # [bsz, z_dim]
        loss = F.mse_loss(instruction_embs, vae_zs)
        result = {'loss': loss}

        return result

if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'instruction_to_vaez', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
    experiments.save_run_data(save_dir, hp)

    model = InstructionToVAEzModel(hp, save_dir)
    model.train_loop()
