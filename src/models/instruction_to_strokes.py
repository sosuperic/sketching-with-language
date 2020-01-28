# instruction_to_strokes.py

"""
Use the annotated MTurk data (ProgressionPairDataset) to train a P(drawing_segment | instruction) model.

Currently, the plan is to use this model to produce the instruction trees (see src/models/segmentation.py).
- Specifically, the greedy parsing will optimize P(segment1 | instruction1) and P(segment2 | instruction2),
where instructions are generated by a P(instruction | drawing_segment) model (trained in src/models/strokes_to_instruction.py)

Note: this is very similar to the sketch_with_plans, except:
    1. Dataset is different
        - We are using ground truth text (ProgressionPairDataset)

Usage:
    PYTHONPATH=. python src/models/instruction_to_strokes.py --cond_instructions initdec
"""

from functools import partial
import os
from os.path import abspath

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import utils
from src.models.core import nn_utils
from src.models.core.train_nn import RUNS_PATH, TrainNN
from src.models.base.instruction_models import (
    ProgressionPairDataset,
    InstructionEncoderTransformer,
    LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH
)
from src.models.base.stroke_models import (
    SketchRNNDecoderGMM
)
from src.models.sketch_rnn import SketchRNNModel
from src.models.sketch_rnn import HParams as SketchRNNHParams

USE_CUDA = torch.cuda.is_available()

class HParams(SketchRNNHParams):
    def __init__(self):
        super().__init__()
        self.cond_instructions = 'initdec'  # 'initdec', 'decinputs'

class InstructionToStrokesModel(SketchRNNModel):
    """"
    SketchRNN that also encodes and conditions on top-level instruction (i.e. instruction for entire
    drawing) generated by an instruction generation model.
    """
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        self.end_epoch_loader = None  # TODO: not generating yet, need to refactor that

        # Model
        # Load text embeddings
        # TODO: move this into some config file
        vocab_size = len(utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH))
        strokes_to_instruction_fp = 'best_models/strokes_to_instruction/catsdecoder-dim_512-model_type_cnn_lstm-use_prestrokes_False/model.pt'
        weights = torch.load(strokes_to_instruction_fp)
        enc_dim = weights['token_embedding.weight'].size(1)
        # enc_dim = hp.enc_dim
        # self.text_embedding = nn.Embedding(vocab_size, hp.enc_dim)
        self.text_embedding = nn.Embedding(vocab_size, enc_dim)  # if we're loading, must be the same size
        self.text_embedding.weight = nn.Parameter(weights['token_embedding.weight'])

        self.enc = InstructionEncoderTransformer(enc_dim, hp.enc_num_layers, hp.dropout, use_categories=False)  # TODO: should this be a hparam
        dec_input_dim = 5 if (hp.cond_instructions == 'initdec') else (5 + enc_dim)  # dec_inputs
        self.dec = SketchRNNDecoderGMM(dec_input_dim, hp.dec_dim, hp.M)  # Method 1 (see one_forward_pass, i.e. decinputs)

        self.models.extend([self.text_embedding, self.enc, self.dec])
        if USE_CUDA:
            for model in self.models:
                model.cuda()

        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def get_data_loader(self, dataset_split, batch_size, categories, max_len, max_per_category, shuffle):
        """
        Args:
            dataset_split (str): 'train', 'valid', 'test'
            batch_size (int)
            categories (str
            shuffle (bool)
        """
        ds = ProgressionPairDataset(dataset_split)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ProgressionPairDataset.collate_fn)
        return loader

    def preprocess_batch_from_data_loader(self, batch):
        """
        Convert tensors to cuda and convert to [len, bsz, ...] instead of [bsz, len, ...]
        """
        preprocessed = []
        for item in batch:
            if type(item) == torch.Tensor:
                item = nn_utils.move_to_cuda(item)
                if item.dim() > 1:
                    item.transpose_(0, 1)
            preprocessed.append(item)
        return preprocessed

    def one_forward_pass(self, batch, average_loss=True):
        """
        Return loss and other items of interest for one forward pass

        Args:
            batch: tuple from DataLoaders
            average_loss (bool): whether to average loss per batch item
                - Current use case: Segmentation model computes loss per segment. Batches
                are a batch of segments for one example.

        Returns:
            dict where 'loss': float Tensor must exist
        """
        strokes, stroke_lens, texts, text_lens, text_indices, cats, cats_idx, urls = batch

        # Create base inputs to decoder
        _, bsz, _ = strokes.size()
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)  # start of sequence
        sos = nn_utils.move_to_cuda(sos)
        dec_inputs = torch.cat([sos, strokes], dim=0)  # add sos at the begining of the strokes; [max_len + 1, bsz, 5]

        #
        # Encode instructions, decode
        # text_indices: [len, bsz], text_lens: [bsz]
        hidden = self.enc(text_indices, text_lens, self.text_embedding,
                            category_embedding=None, categories=cats_idx)  # [bsz, dim]

        # Method 1: concatenate instruction embedding to every time step
        if self.hp.cond_instructions == 'decinputs':
            hidden = hidden.unsqueeze(0)  #  [1, bsz, dim]
            hidden = hidden.repeat(dec_inputs.size(0), 1, 1)  # [max_len + 1, bsz, dim]
            dec_inputs = torch.cat([dec_inputs, hidden], dim=2)  # [max_len + 1, bsz, 5 + dim]
            outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)

        # Method 2: initialize decoder's hidden state with instruction embedding
        elif self.hp.cond_instructions == 'initdec':
            hidden = hidden.unsqueeze(0)  #  [1, bsz, dim]
            hidden_cell = (hidden, hidden.clone())
            outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True, hidden_cell=hidden_cell)

        #
        # Calculate losses
        #
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)
        loss = self.dec.reconstruction_loss(mask,
                                            dx, dy, p,
                                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                            q,
                                            average_loss=average_loss)
        result = {'loss': loss, 'loss_R': loss}

        if ((loss != loss).any() or (loss == float('inf')).any() or (loss == float('-inf')).any()):
            raise Exception('Nan in SketchRNnDecoderGMMOnly forward pass')

        return result

    def generate_and_save(self, data_loader, epoch, n_gens=1, outputs_path=None):
        # TODO: need to overwrite this. SketchRNN's generate_and_save() unpacks a batch from
        # a dataset that returns 4 values. The SketchWithPlansDataset returns 8 values. The
        # encoding of the instructions is different too. Need to refactor that bit to work
        # with both.
        pass

    def inference_pass(self, batch_of_segs, seg_lens, cats_idx):
        pass

if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'instruction_to_strokes', opt.groupname, run_name)
    utils.save_run_data(save_dir, hp)

    model = None
    model = InstructionToStrokesModel(hp, save_dir)
    model.train_loop()
