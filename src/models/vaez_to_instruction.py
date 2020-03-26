"""
Usage:
PYTHONPATH=. python src/models/vaez_to_instruction.py --categories pig
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
from src.models.base.instruction_models import InstructionVAEzDataset, \
    InstructionDecoderLSTM, PAD_ID, SOS_ID, EOS_ID
from src.models.core import experiments, nn_utils
from src.models.core.train_nn import TrainNN
from src.eval.strokes_to_instruction import InstructionScorer

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
        self.enc_dim = 256  # this is the decoder hidden dimension (which is also equal to text embedding dim)
        self.enc_num_layers = 3  # if 0, then just use the z directly (i.e. enc_dim = z_dim)
        self.dec_num_layers = 4
        self.use_categories_dec = True

        self.dropout = 0.1
        # TODO: I need to investigate InstructionDecoderLSTM's layernorm. It has been false mostly.
        self.use_layer_norm = False
        self.rec_dropout = 0.1

        # Keep this fixed
        # self.text_dim = 256
        self.categories_dim = 256
        self.z_dim = 128

        # inference
        self.decode_method = 'greedy'  # 'sample', 'greedy'
        self.tau = 1.0  # sampling text
        self.k = 5      # sampling text

class VAEzToInstructionModel(TrainNN):
    """
    """
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)
        self.tr_loader = self.get_data_loader('train', True)
        self.val_loader = self.get_data_loader('valid', False)
        self.end_epoch_loader = self.get_data_loader('valid', False)

        # Model
        if hp.enc_num_layers == 0:
            hp.enc_dim = hp.z_dim

        text_dim = hp.enc_dim  # text dim must be same as decoder's hidden dimension because it gets directly mapped
        self.text_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, text_dim)
        self.category_embedding = nn.Embedding(35, 	hp.categories_dim)

        # "Encoder" (feed forward that takes z as input)
        self.enc = nn.Sequential(nn.Identity())
        if hp.enc_num_layers > 0:
            self.enc.add_module('fc_0', nn.Linear(hp.z_dim, hp.enc_dim))
            for i in range(hp.enc_num_layers - 1):  # only if it's >= 2
                self.enc.add_module('relu_{}'.format(i+1), nn.ReLU())
                self.enc.add_module('fc_{}'.format(i+1), nn.Linear(hp.enc_dim, hp.enc_dim))

        # Decoder
        dec_input_dim = text_dim + hp.categories_dim
        self.dec = InstructionDecoderLSTM(
                dec_input_dim, hp.enc_dim,
                num_layers=hp.dec_num_layers, dropout=hp.dropout, batch_first=False,
                use_categories=hp.use_categories_dec,
                use_layer_norm=hp.use_layer_norm, rec_dropout=hp.rec_dropout
            )

        self.models = [self.text_embedding, self.category_embedding, self.enc, self.dec]
        for model in self.models:
            model.cuda()

        # For Eval
        self.scorers = [InstructionScorer('rouge'), InstructionScorer('bleu')]

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
        z_emb = self.enc(vae_zs)  # [bsz, enc_dim]

        z_emb = z_emb.unsqueeze(0)  # [1, bsz, enc_dim]
        hidden = z_emb.repeat(self.dec.num_layers, 1, 1)  # [dec_num_layers, bsz, enc_dim]
        cell = z_emb.repeat(self.dec.num_layers, 1, 1)  # [dec_num_layers, bsz, enc_dim]
        # Decode
        texts_emb = self.text_embedding(text_indices)  # [max_text_len, bsz, text_dim]

        logits, _ = self.dec(texts_emb, text_lens,
                             hidden=hidden, cell=cell,
                             token_embedding=self.text_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len, bsz, vocab]; h/c
        loss = self.compute_loss(logits, text_indices, PAD_ID, text_lens=text_lens)
        result = {'loss': loss, 'loss_decode': loss.clone().detach()}

        # import pdb; pdb.set_trace()
        return result


    def compute_loss(self, logits, tf_inputs, pad_id, text_lens=None):
        """
        Args:
            logits: [len, bsz, vocab]
            tf_inputs: [len, bsz] ("teacher-forced inputs", inputs to decoder used to generate logits)
                (text_indices_w_sos_eos)
            pad_id (int)
            text_lens (list of ints): used when use_layer_norm=True
                The Haste LayerNormLSTM I'm using doesn't take packed sequences, so have to take care of masking here
        """
        logits = logits[:-1, :, :]    # last input that produced logits is EOS. Don't care about the EOS -> mapping
        targets = tf_inputs[1: :, :]  # remove first input (sos)

        max_len, bsz, vocab_size = logits.size()

        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=pad_id)

        return loss

    ####################################################################
    #
    # INFERENCE
    #
    ####################################################################
    def end_of_epoch_hook(self, data_loader, epoch, outputs_path=None, writer=None):
        """
        Args:
            data_loader: DataLoader
            epoch: int
            outputs_path: str
            writer: Tensorboard Writer
        """
        for model in self.models:
            model.eval()

        with torch.no_grad():
            # Generate texts on validation set
            inference = self.inference_loop(data_loader, writer=writer, epoch=epoch)
            out_fp = os.path.join(outputs_path, 'samples_e{}.json'.format(epoch))
            utils.save_file(inference, out_fp, verbose=True)

    def inference_loop(self, loader, writer=None, epoch=None):
        """
        Args:
            loader: DataLoader
            writer: Tensorboard writer (used during validation)
            epoch: int (used for writer)

        Returns: list of dicts
        """
        inference = []
        for i, batch in enumerate(loader):
            texts, text_lens, text_indices, cats, cats_idx, vae_zs = batch
            decoded_probs, decoded_ids, decoded_texts = self.inference_pass(vae_zs, cats_idx)

            for j, instruction in enumerate(texts):
                gt, gen = instruction, decoded_texts[j]

                # construct results
                result = {
                    'ground_truth': gt,
                    'generated': gen,
                    'category': cats[j],
                }
                for scorer in self.scorers:
                    for name, value in scorer.score(gt, gen).items():
                        result[name] = value
                        if writer:
                            writer.add_scalar('inference/{}'.format(name), value, epoch * self.val_loader.__len__() + i)
                inference.append(result)

                # log
                text = 'Ground truth: {}  \n  \nGenerated: {}'.format(
                    instruction, decoded_texts[j])
                if writer:
                    writer.add_text('inference/sample', text, epoch * self.val_loader.__len__() + i)

        return inference

    def inference_pass(self, vae_zs, cats_idx):
        """

        Args:
            vae_zs: [bsz, z_dim]
            cats_idx: [bsz] LongTensor

        Returns:
            decoded_probs: [bsz, max_len, vocab]
            decoded_ids: [bsz, max_len]
            decoded_texts: list of strs
        """
        bsz = vae_zs.size(0)

        # Get hidden and cell by encoding z
        z_emb = self.enc(vae_zs)  # [bsz, enc_dim]
        z_emb = z_emb.unsqueeze(0)  # [1, bsz, enc_dim]
        hidden = z_emb.repeat(self.dec.num_layers, 1, 1)  # [dec_num_layers, bsz, enc_dim]
        cell = z_emb.repeat(self.dec.num_layers, 1, 1)  # [dec_num_layers, bsz, enc_dim]

        # Create init input
        init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
        init_ids.transpose_(0, 1)  # [1, bsz]

        decoded_probs, decoded_ids, decoded_texts = self.dec.generate(
            self.text_embedding,
            category_embedding=self.category_embedding, categories=cats_idx,
            init_ids=init_ids, hidden=hidden, cell=cell,
            pad_id=PAD_ID, eos_id=EOS_ID, max_len=200,  # TODO: set max_len to max_len on data
            decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
            idx2token=self.tr_loader.dataset.idx2token
        )

        return decoded_probs, decoded_ids, decoded_texts


if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'vaez_to_instruction', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
    experiments.save_run_data(save_dir, hp)

    model = VAEzToInstructionModel(hp, save_dir)
    model.train_loop()
