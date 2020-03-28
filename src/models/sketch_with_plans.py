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
from src.data_manager.quickdraw import save_strokes_as_img, save_multiple_strokes_as_img
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
        self.instruction_set = 'toplevel'  # 'toplevel_leaves',  'leaves', 'stack'
        self.dec_dim = 2048
        self.lr = 0.0001

        self.use_layer_norm = True
        self.dropout = 0.1
        self.rec_dropout = 0.1

        self.freeze_enc = False

        # do not change these
        self.use_categories_dec = True
        self.cond_instructions = 'match'  # 'match', 'decinputs'
        self.categories_dim = 256
        self.loss_match = 'triplet' # when cond_instructions==match, 'triplet', 'decode'
        self.enc_num_layers = 1
        self.enc_dim = 512 # only used with loss_match=triplet or cond_instructions=decinputs

class SketchRNNWithPlans(SketchRNNModel):
    """"
    SketchRNN that also encodes and conditions on top-level instruction (i.e. instruction for entire
    drawing) generated by an instruction generation model.
    """
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir, skip_data=True)

        self.tr_loader = self.get_data_loader('train', True, self.hp.batch_size)
        self.val_loader = self.get_data_loader('valid', False, self.hp.batch_size)
        self.end_epoch_loader = self.get_data_loader('valid', False, batch_size=1)

        #
        # Model
        #

        # text and category embeddings
        self.text_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.enc_dim)
        self.models.append(self.text_embedding)
        self.category_embedding = None
        if hp.use_categories_dec:
            self.category_embedding = nn.Embedding(35, 	hp.categories_dim)
            self.models.append(self.category_embedding)

        # decoder
        dec_input_dim = 5
        if self.category_embedding:
            dec_input_dim += hp.categories_dim
        if hp.cond_instructions == 'decinputs':
            dec_input_dim += hp.enc_dim
        self.dec = SketchRNNDecoderGMM(dec_input_dim, hp.dec_dim, hp.M,
            dropout=hp.dropout, use_layer_norm=hp.use_layer_norm, rec_dropout=hp.rec_dropout)
        self.models.append(self.dec)

        # For shaping representation loss, want to decode into instructions
        if hp.cond_instructions == 'decinputs':
            self.enc = InstructionEncoderTransformer(hp.enc_dim, hp.enc_num_layers, hp.dropout,
                use_categories=True, categories_dim=hp.categories_dim)
            self.models.append(self.enc)
        elif hp.cond_instructions == 'match':
            if hp.loss_match == 'triplet':
                # TODO: use category embedding? (see decinputs)
                self.enc = InstructionEncoderTransformer(hp.enc_dim, hp.enc_num_layers, hp.dropout, use_categories=False)  # TODO: should this be a hparam
                self.fc_dec = nn.Linear(hp.dec_dim, hp.enc_dim)  # project decoder hidden states to enc hidden states
                self.models.extend([self.enc, self.fc_dec])
            elif hp.loss_match == 'decode':
                ins_hid_dim = hp.enc_dim  # Note: this has to be because there's no fc_out layer. I just multiply by token embedding directly to get outputs
                self.ins_dec = InstructionDecoderLSTM(
                    hp.enc_dim + hp.categories_dim, ins_hid_dim, num_layers=4, dropout=0.1, batch_first=False,
                    condition_on_hc=False, use_categories=True)
                self.models.append(self.ins_dec)

        if USE_CUDA:
            for model in self.models:
                model.cuda()

        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    def get_data_loader(self, dataset_split, shuffle, batch_size):
        """
        Args:
            dataset_split (str): 'train', 'valid', 'test'
            shuffle (bool)
        """
        if self.hp.instruction_set in ['toplevel', 'toplevel_leaves', 'leaves']:
            ds = SketchWithPlansConditionEntireDrawingDataset(dataset=self.hp.dataset,
                                                              max_len=self.hp.max_len,
                                                              categories=self.hp.categories,
                                                              max_per_category=self.hp.max_per_category,
                                                              dataset_split=dataset_split,
                                                              instruction_set=self.hp.instruction_set,
                                                              prob_threshold=self.hp.prob_threshold)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                collate_fn=ProgressionPairDataset.collate_fn)
        elif self.hp.instruction_set in ['stack', 'stack_leaves']:
            ds = SketchWithPlansConditionSegmentsDataset(dataset=self.hp.dataset,
                                                         max_len=self.hp.max_len,
                                                         categories=self.hp.categories,
                                                         max_per_category=self.hp.hp.max_per_category,
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
        if self.hp.instruction_set in ['stack', 'stack_leaves']:
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

        elif self.hp.instruction_set in ['toplevel', 'toplevel_leaves', 'leaves']:
            # triplet loss
            if self.hp.cond_instructions == 'decinputs':  # concatenate instruction embedding to every time step
                # Encode instructions
                # text_indices: [len, bsz], text_lens: [bsz]
                instructions_emb = self.enc(
                    text_indices, text_lens, self.text_embedding,
                    category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, enc_dim]

                # decode
                instructions_emb = instructions_emb.unsqueeze(0)  #  [1, bsz, dim]
                instructions_emb = instructions_emb.repeat(dec_inputs.size(0), 1, 1)  # [max_len + 1, bsz, dim]
                dec_inputs = torch.cat([dec_inputs, instructions_emb], dim=2)  # [max_len + 1, bsz, inp_dim]
                outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)

            elif self.hp.cond_instructions == 'match':  # match decoder's hidden representations to encoded language
                # decode
                outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.dec(dec_inputs, output_all=True)  # outputs: [max_seq_len, bsz, dim]
                outputs = outputs.mean(dim=0)  # [bsz, dec_dim]

                if self.hp.loss_match == 'triplet':
                    # Encode instructions
                    # text_indices: [len, bsz], text_lens: [bsz]
                    instructions_emb = self.enc(
                        text_indices, text_lens, self.text_embedding,
                        category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, enc_dim]
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
        # Calculate reconstruction and final loss
        #
        mask, dx, dy, p = self.dec.make_target(strokes, stroke_lens, self.hp.M)

        loss_R = self.dec.reconstruction_loss(mask,
                                              dx, dy, p,
                                              pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
                                              q)

        if self.hp.cond_instructions == 'decinputs':
            loss = loss_R
            result = {'loss': loss, 'loss_R': loss_R.clone().detach()}
        elif self.hp.cond_instructions == 'match':
            loss = loss_R + loss_match
            result = {'loss': loss, 'loss_R': loss_R.clone().detach(),
                      'loss_match': loss_match.clone().detach()}

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

    ##############################################################################
    # Generate
    ##############################################################################
    def end_of_epoch_hook(self, data_loader, epoch, outputs_path=None, writer=None):
          # match not implemented / tested yet
        if (self.hp.cond_instructions == 'decinputs') and \
            (self.hp.instruction_set in ['toplevel', 'toplevel_leaves', 'leaves']):
            self.generate_and_save(data_loader, epoch, 25, outputs_path=outputs_path)

    def generate_and_save(self, data_loader, epoch, n_gens, outputs_path=None):
        """
        Generate and save drawings
        """
        n = 0
        gen_strokes = []
        gt_strokes = []
        gt_texts = []
        for i, batch in enumerate(data_loader):
            batch = self.preprocess_batch_from_data_loader(batch)
            strokes, stroke_lens, texts, text_lens, text_indices, cats, cats_idx, urls = batch

            max_len, bsz, _ = strokes.size()

            if self.hp.cond_instructions == 'decinputs':
                # Encode instructions
                # text_indices: [len, bsz], text_lens: [bsz]
                instructions_emb = self.enc(
                    text_indices, text_lens, self.text_embedding,
                    category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, enc_dim]
                z = instructions_emb

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
                if self.hp.cond_instructions == 'decinputs':  # input is last state, z, and hidden_cell
                    input = torch.cat([s, z.unsqueeze(0)], dim=2)  # [1 (len), 1 (bsz), input_dim (5) + z_dim (128)]

                elif self.hp.cond_instructions == 'match':  # input is last state and hidden_cell
                    input = s   # [1, bsz (1), 5]

                if self.hp.use_categories_dec \
                    and hasattr(self, 'category_embedding'):
                    # hack because VAE was trained with use_categories_dec=True but didn't actually have a category embedding
                    cat_embs = self.category_embedding(cats_idx)  # [bsz (1), cat_dim]
                    input = torch.cat([input, cat_embs.unsqueeze(0)], dim=2)  # [1, 1, dim]
                    # dim = 5 + cat_dim if decodergmm, 5 + z_dim + cat_dim if vae

                outputs, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = \
                    self.dec(input, stroke_lens=stroke_lens, output_all=False, hidden_cell=hidden_cell)
                hidden_cell = (hidden, cell)  # for next timee step
                # sample next state
                s, dx, dy, pen_up, eos = self.sample_next_state(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)

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
            # output_fp = os.path.join(outputs_path, f'e{epoch}-gen{n}.jpg')
            # save_strokes_as_img(sequence, output_fp)

            # Save original as well
            output_fp = os.path.join(outputs_path, f'e{epoch}-gt{n}.jpg')
            strokes_x = strokes[:, 0,
                        0]  # first 0 for x because sample_next_state etc. only using 0-th batch item; 2nd 0 for dx
            strokes_y = strokes[:, 0, 1]  # 1 for dy
            strokes_x = np.cumsum(strokes_x.cpu().numpy())
            strokes_y = np.cumsum(strokes_y.cpu().numpy())
            strokes_pen = strokes[:, 0, 3].cpu().numpy()
            strokes_out = np.stack([strokes_x, strokes_y, strokes_pen]).T
            # save_strokes_as_img(strokes_out, output_fp)

            gen_strokes.append(sequence)
            gt_strokes.append(strokes_out)
            gt_texts.append(texts[0])  # 0 because batch size is 1

            n += 1
            if n == n_gens:
                break

        # save grid drawings
        rowcol_size = 5
        chunk_size = rowcol_size ** 2
        for i in range(0, chunk_size, len(gen_strokes)):
            output_fp = os.path.join(outputs_path, f'e{epoch}_gen{i}-{i+chunk_size}.jpg')
            save_multiple_strokes_as_img(gen_strokes[i:i+chunk_size], output_fp)

            output_fp = os.path.join(outputs_path, f'e{epoch}_gt{i}-{i+chunk_size}.jpg')
            save_multiple_strokes_as_img(gt_strokes[i:i+chunk_size], output_fp)

            # save texts
            output_fp = os.path.join(outputs_path, f'e{epoch}_texts{i}-{i+chunk_size}.json')
            utils.save_file(gt_texts[i:i+chunk_size], output_fp)

    ##############################################################################
    # Load
    ##############################################################################
    def load_enc_and_catemb_from_pretrained(self, dir):
        """
        Copy pretrained encoder and category embedding weights. They must
        be the same dimension as self's dimension.s.

        Args:
            dir: str
        """
        print('Loading encoder and category embedidng weights from pretrained: ', dir)
        fp = os.path.join(dir, 'model.pt')
        trained_state_dict = torch.load(fp)
        self_state_dict = self.state_dict()

        for name, param in trained_state_dict.items():
            if name.startswith('enc') or name.startswith('category_embedding'):
                self_state_dict[name].copy_(param)


if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    parser.add_argument('--load_enc_and_catemb', default=None, help='Directory to load trained instruction encoder')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'sketchwplans', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
    experiments.save_run_data(save_dir, hp)

    model = SketchRNNWithPlans(hp, save_dir)
    if opt.load_enc_and_catemb:
        model.load_enc_and_catemb_from_pretrained(opt.load_enc_and_catemb)

    if hp.freeze_enc:
        for param in model.enc.parameters():
            param.requires_grad = False

    model.train_loop()
