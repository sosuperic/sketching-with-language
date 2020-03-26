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
        self.max_per_category = 70000
        self.prob_threshold = 0.0  # prune trees
        self.batch_size = 128
        self.lr = 0.0001

        # Model
        self.enc_dim = 512
        self.enc_num_layers = 1
        self.dropout = 0.1

        # Keep this fixed
        self.categories_dim = 256
        self.enc_dim = 128  # has to be same as z_dim

class InstructionToVAEzModel(TrainNN):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)
        self.tr_loader = self.get_data_loader('train', hp.batch_size, hp.max_per_category, True)
        self.val_loader = self.get_data_loader('valid', hp.batch_size, hp.max_per_category, False)
        self.end_epoch_loader = None

        # Model
        self.text_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.enc_dim)
        self.category_embedding = nn.Embedding(35, 	hp.categories_dim)
        self.enc = InstructionEncoderTransformer(hp.enc_dim, hp.enc_num_layers, hp.dropout, use_categories=True)

        for model in [self.text_embedding, self.category_embedding, self.enc]:
            model.cuda()

    def get_data_loader(self, dataset_split, batch_size, max_per_category, shuffle):
        ds = InstructionVAEzDataset(dataset_split=dataset_split, max_per_category=max_per_category)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
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
        texts, text_lens, batch_text_indices, cats, cats_idx, vae_zs = batch
        instruction_embs = self.enc(texts, text_lens, self.text_embedding,
                                    category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, enc_dim]
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
