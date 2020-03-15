# sketch_with_plans.py

"""
Usage:
    PYTHONPATH=. python src/models/sketch_with_plans.py --instruction_set toplevel_leaves
    PYTHONPATH=. python src/models/sketch_with_plans.py --dataset ndjson --instruction_set stack
"""

from datetime import datetime
from functools import partial
import os
from os.path import abspath
import random

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import RUNS_PATH
from src import utils
from src.models.core import experiments, nn_utils
from src.models.core.train_nn import TrainNN
from src.models.base.instruction_models import ProgressionPairDataset, \
    SketchWithPlansConditionEntireDrawingDataset, \
    SketchWithPlansConditionSegmentsDataset, \
    InstructionEncoderTransformer, \
    InstructionDecoderLSTM
from src.models.base.stroke_models import SketchRNNDecoderGMM
from src.models.sketch_rnn import SketchRNNModel
from src.models.sketch_rnn import HParams as SketchRNNHParams

USE_CUDA = torch.cuda.is_available()

class HParams(SketchRNNHParams):
    def __init__(self):
        super().__init__()

        # Data
        self.dataset = 'ndjson'   # 'progressionpair' or 'ndjson'
        self.max_per_category = 2000
        self.prob_threshold = 0.0  # prune trees

        # Model
        self.instruction_set = 'toplevel'  # 'toplevel_leaves',  'stack'
        self.dec_dim = 2048
        self.lr = 0.0001

        # do not change these
        self.use_categories_dec = True
        self.cond_instructions = 'match'  # 'match'
        self.categories_dim = 256
        self.loss_match = 'triplet' # 'triplet', 'decode'
        self.enc_num_layers = 1
        self.enc_dim = 512 # only used with loss_match=triplet

class SketchRNNWithPlans(SketchRNNModel):
    """"
    SketchRNN that also encodes and conditions on top-level instruction (i.e. instruction for entire
    drawing) generated by an instruction generation model.
    """
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        self.end_epoch_loader = None  # TODO: not generating yet, need to refactor that

        # Model
        self.text_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.enc_dim)

        self.category_embedding = None
        if hp.use_categories_dec:
            self.category_embedding = nn.Embedding(35, 	hp.categories_dim)
            self.models.append(self.category_embedding)
        dec_input_dim = (5 + hp.categories_dim) if self.category_embedding else 5
        self.dec = SketchRNNDecoderGMM(dec_input_dim, hp.dec_dim, hp.M)  # Method 1 (see one_forward_pass, i.e. decinputs)

        self.models.extend([self.text_embedding,  self.dec])

        # For shaping representation loss, want to decode into instructions
        if hp.loss_match == 'triplet':
            self.enc = InstructionEncoderTransformer(hp.enc_dim, hp.enc_num_layers, hp.dropout, use_categories=False)  # TODO: should this be a hparam
            self.fc_dec = nn.Linear(hp.dec_dim, hp.enc_dim)  # project decoder hidden states to enc hidden states
            self.models.extend([self.enc, self.fc_dec])
        elif hp.loss_match == 'decode':
            ins_hid_dim = hp.enc_dim  # Note: this has to be because there's no fc_out layer. I just multiply by token embedding directly to get outputs
            self.ins_dec = InstructionDecoderLSTM(
                hp.enc_dim + hp.categories_dim, ins_hid_dim, num_layers=4, dropout=0.1, batch_first=False,
                condition_on_hc=False, use_categories=True)
            self.models.extend([self.ins_dec])

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
        if self.hp.instruction_set in ['toplevel', 'toplevel_leaves']:
            ds = SketchWithPlansConditionEntireDrawingDataset(dataset=hp.dataset,
                                                              max_len=max_len,
                                                              max_per_category=hp.max_per_category,
                                                              dataset_split=dataset_split,
                                                              instruction_set=self.hp.instruction_set,
                                                         prob_threshold=self.hp.prob_threshold)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=ProgressionPairDataset.collate_fn)
        elif self.hp.instruction_set == 'stack':
            ds = SketchWithPlansConditionSegmentsDataset(dataset=hp.dataset,
                                                         max_len=max_len,
                                                         max_per_category=hp.max_per_category,
                                                         dataset_split=dataset_split,
                                                         instruction_set=self.hp.instruction_set,
                                                         prob_threshold=self.hp.prob_threshold)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=partial(SketchWithPlansConditionSegmentsDataset.collate_fn, token2idx=ds.token2idx))
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

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        Args:
            batch: tuple from DataLoaders

        Returns:
            dict where 'loss': float Tensor must exist
        """
        strokes, stroke_lens, texts, text_lens, text_indices, cats, cats_idx, urls = batch
        # batch is 1st dimension (not 0th) due to preprocess_batch()

        # Create base inputs to decoder
        _, bsz, _ = strokes.size()
        sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * bsz).unsqueeze(0)  # start of sequence
        sos = nn_utils.move_to_cuda(sos)
        dec_inputs = torch.cat([sos, strokes], dim=0)  # add sos at the begining of the strokes; [max_len + 1, bsz, 5]

        if self.hp.use_categories_dec:
            cat_embs = self.category_embedding(cats_idx)  # [bsz, cat_dim]
            cat_embs = cat_embs.repeat(dec_inputs.size(0), 1, 1)  # [max_len + 1, bsz, cat_dim]
            dec_inputs = torch.cat([dec_inputs, cat_embs], dim=2)  # [max_len+1, bsz, 5 + cat_dim]

        #
        # Encode instructions, decode
        #
        if self.hp.instruction_set == 'stack':
            # text_indices: [max_seq_len, bsz, max_instruction_len], # text_lens: [max_seq_len, bsz]

            # decoder's hidden states are "matched" with language representations
            outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)

            # # PLAN get hidden states and instruction stack for each batch input, for each segment
            # import pdb; pdb.set_trace()  # outputs: [max_seq_len, bsz, dim]
            # loss_match = 0.0

            # For each sequence in batch, divide drawing into segments (based on penup strokes)
            # For each segment, compute a matching loss between its hidden state and the
            # the encoded instruction stack for that segment
            all_instruction_embs = []
            all_seg_hiddens = []
            for i in range(bsz):
                penups = np.where(strokes[:,i,:].cpu().numpy()[:,3] == 1)[0].tolist()
                penups = ([0] + penups) if (penups[0] != 0) else penups  # first element could already be 0
                # TODO: find other place that I do [0] + penups, see if I need to account for the case
                # where the first element is 0

                # Encode instruction stacks
                # text_indices: [max_seq_len, bsz, max_instruction_len]
                instructions = [text_indices[start_idx, i, :] for start_idx in penups[:-1]]
                # Note on above:
                #   [:-1] because that's the end of the last segment
                #   instructions for each timestep within segment are the same, take the start_idx
                instructions = torch.stack(instructions, dim=1)  # [max_instruction_len, n_segs] (max across all segs in batch)
                # (n_segs is the "batch" for the encoder)
                instructions_lens = [text_lens[start_idx, i].item() for start_idx in penups[:-1]]
                instructions = instructions[:max(instructions_lens), :]  # encoder requires this
                cur_cats_idx = [cats_idx[i] for _ in range(len(instructions_lens))]  # all segs are from same drawing (i.e. same category)
                instruction_embs = self.enc(instructions, instructions_lens, self.text_embedding,
                                            category_embedding=None, categories=cur_cats_idx)  # [n_segs, dim]
                all_instruction_embs.append(instruction_embs)

                # Compute hidden states mean for each seg
                seg_hiddens = []
                for j in range(len(penups) - 1):  # n_segs
                    start_idx = penups[j]
                    end_idx = penups[j+1]
                    seg_outputs = outputs[start_idx:end_idx+1, i, :]  # [seg_len, dim]
                    seg_hidden = seg_outputs.mean(dim=0)  # [dim]
                    seg_hiddens.append(seg_hidden)
                seg_hiddens = torch.stack(seg_hiddens, dim=0)  # [n_segs, dim]
                all_seg_hiddens.append(seg_hiddens)

            # Concate all segs across all batch items
            if self.hp.loss_match == 'triplet':
                all_instruction_embs = torch.cat(all_instruction_embs, dim=0)  # [n_total_segs, enc_dim]
                all_seg_hiddens = torch.cat(all_seg_hiddens, dim=0)  # [n_total_segs, dec_dim]
                all_seg_hiddens = self.fc_dec(all_seg_hiddens)  # [n_total_segs, enc_dim]

                # Compute triplet loss
                pos = (all_seg_hiddens - all_instruction_embs) ** 2  # [n_total_segs, enc_dim]
                all_instruction_embs_shuffled = all_instruction_embs[torch.randperm(pos.size(0)), :]  # [n_total_segs, enc_dim]
                neg = (all_seg_hiddens - all_instruction_embs_shuffled) ** 2  # [n_total_segs, enc_dim]
                loss_match = (pos - neg).mean() + torch.tensor(0.1).to(pos.device)  # positive - negative + alpha
                loss_match = max(torch.tensor(0.0), loss_match)
            elif self.hp.loss_match == 'decode':
                raise NotImplementedError

            # TODO: check if text_indices is correct

        elif self.hp.instruction_set in ['toplevel', 'toplevel_leaves']:
            outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)  # outputs: [max_seq_len, bsz, dim]
            outputs = outputs.mean(dim=0)  # [bsz, dec_dim]

            # triplet loss
            if self.hp.loss_match == 'triplet':
                # Encode instructions
                # text_indices: [len, bsz], text_lens: [bsz]
                instructions_emb = self.enc(text_indices, text_lens, self.text_embedding,
                                category_embedding=None, categories=cats_idx)  # [bsz, enc_dim]
                outputs = self.fc_dec(outputs)  # [bsz, enc_dim]

                pos = (outputs - instructions_emb) ** 2  # [bsz, enc_dim]
                instructions_emb_shuffled = instructions_emb[torch.randperm(bsz), :]  # [bsz, enc_dim]
                neg = (outputs - instructions_emb_shuffled) ** 2  # [bsz, enc_dim]
                loss_match = (pos - neg).mean() + torch.tensor(0.1).to(pos.device)  # positive - negative + alpha
                loss_match = max(torch.tensor(0.0), loss_match)
            elif self.hp.loss_match == 'decode':
                hidden = nn_utils.move_to_cuda(torch.zeros(self.ins_dec.num_layers, bsz, self.ins_dec.hidden_dim))
                cell = nn_utils.move_to_cuda(torch.zeros(self.ins_dec.num_layers, bsz, self.ins_dec.hidden_dim))

                # Decode
                texts_emb = self.text_embedding(text_indices)  # [len, bsz, dim]
                logits, texts_hidden = self.ins_dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                                                    token_embedding=self.text_embedding,
                                                    category_embedding=self.category_embedding, categories=cats_idx)
                loss_match = self.compute_dec_loss(logits, text_indices)

        #
        # Calculate losses
        #
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)

        loss_R = self.dec.reconstruction_loss(mask,
                                            dx, dy, p,
                                            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                            q)
        loss = loss_R + loss_match
        result = {'loss': loss, 'loss_R': loss_R, 'loss_match': loss_match}

        if ((loss != loss).any() or (loss == float('inf')).any() or (loss == float('-inf')).any()):
            raise Exception('Nan in SketchRNnDecoderGMMOnly forward pass')

        return result


    def compute_dec_loss(self, logits, tf_inputs):
        """
        Args:
            logits: [len, bsz, vocab]
            tf_inputs: [len, bsz] ("teacher-forced inputs", inputs to decoder used to generate logits)
                (text_indices_w_sos_eos)
        """
        logits = logits[:-1, :, :]    # last input that produced logits is EOS. Don't care about the EOS -> mapping
        targets = tf_inputs[1: :, :]  # remove first input (sos)

        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=0)

        return loss

    def generate_and_save(self, data_loader, epoch, n_gens=1, outputs_path=None):
        # TODO: need to overwrite this. SketchRNN's generate_and_save() unpacks a batch from
        # a dataset that returns 4 values. The SketchWithPlansDataset returns 8 values. The
        # encoding of the instructions is different too. Need to refactor that bit to work
        # with both.
        pass

if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'sketchwplans', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
    experiments.save_run_data(save_dir, hp)

    model = None
    model = SketchRNNWithPlans(hp, save_dir)
    model.train_loop()
