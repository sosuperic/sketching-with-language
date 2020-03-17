"""
Usage:
PYTHONPATH=. python src/models/retrieve_refine.py
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
from src.data_manager.quickdraw import build_category_index_nodata
# from src.data_manager.quickdraw import save_strokes_as_img, save_multiple_strokes_as_img
from src.models.base.instruction_models import ProgressionPairDataset
# from src.models.base.stroke_models import (NdjsonStrokeDataset,
#                                            NpzStrokeDataset,
#                                            SketchRNNDecoderGMM,
#                                            SketchRNNDecoderLSTM,
#                                            SketchRNNVAEEncoder)

from src.models.core import experiments, nn_utils
from src.models.core.train_nn import TrainNN

# USE_CUDA = torch.cuda.is_available()


from src.models.sketch_rnn import HParams as SketchRNNHParams
from src.models.sketch_rnn import SketchRNNVAEModel#, SketchRNNDecoderGMM


class HParams(SketchRNNHParams):
    def __init__(self):
        super().__init__()

        # This is same as in SketchRNNHParams, just copying over here for clarity
        self.dataset = 'ndjson'  # TODO:
        self.categories = 'all'
        self.max_per_category = 70000
        self.batch_size = 64

        self.model_type = 'retrieverefine'
        self.fixed_mem = True
        self.sel_k = 3


class SketchRetrieveRefineModel(SketchRNNVAEModel):
    """
    Create entire sketch model by combining encoder and decoder. Training, generation, etc.
    """
    def __init__(self, hp, save_dir, skip_data=False):
        super().__init__(hp, save_dir, skip_data=skip_data)

        # What else to do here?
        # 1. Load retrieval set
        self.idx2cat, self.cat2idx = build_category_index_nodata()
        self.cat_to_retrieval_text = defaultdict(list)
        # self.cat_to_retrieval_vals = defaultdict(list)
        max_len = 0
        # n = 0
        self.n_ret_per_cat = 251
        self.retrieval_vals = np.zeros((201, 35 * self.n_ret_per_cat, 5))
        for split in ['train', 'valid', 'test']:
            ds = ProgressionPairDataset(split)
            for i in range(len(ds)):
                item = ds.__getitem__(i)
                stroke5, text, category = item[0], item[2], item[4]
                # import pdb; pdb.set_trace()
                cat_idx = self.cat2idx[category]
                max_len = max(stroke5.shape[0], max_len)
                self.cat_to_retrieval_text[category].append(text)
                # self.cat_to_retrieval_vals[category].append(stroke5)
                n_cats = len(self.cat_to_retrieval_text[category])
                # if n_cats > 250:
                # print(category, cat_idx, n_cats)
                self.retrieval_vals[:len(stroke5), cat_idx * self.n_ret_per_cat + n_cats, :] = stroke5
                # n += 1
        # print(max_len)
        self.retrieval_vals = self.retrieval_vals[:max_len, :, :]
        self.retrieval_vals = nn_utils.move_to_cuda(torch.Tensor(self.retrieval_vals))
        if self.hp.fixed_mem:
            self.retrieval_vals.requires_grad = False
        else:
            self.retrieval_vals.requires_grad = True

        self.query_ff = nn.Sequential(
            nn.Linear(self.hp.z_dim + 1, self.hp.z_dim),
            nn.ReLU(),
            nn.Linear(self.hp.z_dim, self.n_ret_per_cat)
        )
        self.query_ff.cuda()

        self.optimizers.append(optim.Adam(self.query_ff.parameters(), hp.lr))

    def retrieve(self, strokes, cats, cats_idx):
        z, mu, sigma_hat = self.enc(strokes)  # [bsz, z_dim]
        query = z
        query = torch.cat([query, cats_idx.unsqueeze(-1).float()], dim=1)  # [bsz, z_dim + 1]
        query = self.query_ff(query)  # [bsz, 250]

        topk_vals, topk_idxs = query.topk(self.hp.sel_k)  # [bsz, sel_k], [bsz, sel_k]
        topk_vals = F.softmax(topk_vals, dim=1)  # [bsz, sel_k]

        all_ret_vals = []
        bsz = query.size(0)
        for i in range(bsz):  # bsz
            cat_idx = cats_idx[i]
            cat_vals = self.retrieval_vals[cat_idx * self.n_ret_per_cat: (cat_idx + 1) * self.n_ret_per_cat,:,:]  # [max_len, n_ret_per_cat, 5]
            ret_vals = torch.index_select(cat_vals, 1, topk_idxs[i])  # [max_len, sel_k, 5]
            if not self.hp.fixed_mem:
                ret_vals.requires_grad = True
            all_ret_vals.append(ret_vals)
        all_ret_vals = torch.stack(all_ret_vals)  # [bsz, max_len, sel_k, 5]
        all_ret_vals = all_ret_vals.transpose(0,1).reshape(-1, bsz * self.hp.sel_k, 5)  # [max_len, bsz * sel_k, 5]

        all_ret_vals_z, all_ret_vals_mu, all_ret_vals_sigmahat = self.enc(all_ret_vals)  # [bsz * sel_k, z_dim], ..., ...

        # TODO: do the above (result of self.enc have grad? Yes. all_ret_vals does not though)
        # import pdb; pdb.set_trace()

        all_ret_vals_z = all_ret_vals_z.view(bsz, self.hp.sel_k, self.hp.z_dim)  # [bsz, sel_k, z_dim]
        all_ret_vals_mu = all_ret_vals_mu.view(bsz, self.hp.sel_k, self.hp.z_dim)  # [bsz, sel_k, z_dim]
        all_ret_vals_sigmahat = all_ret_vals_sigmahat.view(bsz, self.hp.sel_k, self.hp.z_dim)  # [bsz, sel_k, z_dim]

        topk_vals = topk_vals.unsqueeze(-1)
        ret_z = topk_vals * all_ret_vals_z  # [bsz, sel_k, z_dim]
        ret_mu = topk_vals * all_ret_vals_mu  # [bsz, sel_k, z_dim]
        ret_sigmahat = topk_vals * all_ret_vals_sigmahat  # [bsz, sel_k, z_dim]
        ret_z = ret_z.mean(dim=1)  # [bsz, z_dim]
        ret_mu = ret_mu.mean(dim=1)  # [bsz, z_dim]
        ret_sigmahat = ret_sigmahat.mean(dim=1)  # [bsz, z_dim]

        return ret_z, ret_mu, ret_sigmahat

        # all_strokes = []
        # max_len = 0
        # for cat in cats:
        #     keys, vals = self.cat_to_retrieval_keys[cat], self.cat_to_retrieval_vals[cat]

        #     # TODO: right now this is just a hack, selecting random elements...
        #     random.shuffle(keys)
        #     random.shuffle(vals)
        #     vals = vals[:self.hp.sel_k]

        #     strokes = np.vstack(vals)  # [len, 5]
        #     all_strokes.append(strokes)
        #     max_len = max(max_len, strokes.shape[0])

        # retrieved = np.zeros((max_len, len(cats), 5))  # [max_len, bsz, 5]
        # for i, strokes in enumerate(all_strokes):
        #     retrieved[:len(strokes),i,:] = strokes

        # retrieved = nn_utils.move_to_cuda(torch.FloatTensor(retrieved))

        # return retrieved

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
        # These two lines are different
        ret_z, ret_mu, ret_sigmahat = self.retrieve(strokes, cats, cats_idx)  # [bsz, enc_dim]
        z, mu, sigma_hat = ret_z, ret_mu, ret_sigmahat
        # z, mu, sigma_hat = self.enc(retrieved_enc)  # each [bsz, z_dim]


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



if __name__ == "__main__":
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--load_model_path', help='path to directory containing model to load for inference')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'retrieve_refine', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)

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
    if hp.model_type == 'retrieverefine':
        model = SketchRetrieveRefineModel(hp, save_dir, skip_data=opt.inference)

    if opt.inference:
        model.load_model(opt.load_model_path)
        setattr(model.hp, 'temperature', temp)  # this may vary at inference time
        model.save_imgs_inference_time(opt.load_model_path)
    else:
        model.train_loop()
