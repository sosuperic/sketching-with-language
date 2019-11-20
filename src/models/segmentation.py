# segmentation.py

"""
Currently uses trained Stroke2Instruction model to segment unseen sequences.
"""

import argparse
import numpy as np
import os
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import utils
from src.data_manager.quickdraw import QUICKDRAW_DATA_PATH, final_categories, convert_stroke5_to_ndjson_seq, \
    create_progression_image_from_ndjson_seq
from src.models.base.stroke_models import StrokeDataset
from src.models.core import nn_utils
from src.models.instruction_gen import StrokeToInstructionModel, EOS_ID



SEGMENTATIONS_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'segmentations')

##############################################################################
#
# Hyperparameters
#
##############################################################################
class HParams():
    def __init__(self):
        self.max_k = None  # for DP method; int or None
        self.notes = ''

        # TODO: these are only necessary because we instantiate a StrokeToInstructionModel...
        # which loads the ProgressionPairsDataset. Should refactor
        # self.batch_size = 1
        # self.use_prestrokes = False

##############################################################################
#
# Model
#
##############################################################################

class SegmentationModel(object):
    def __init__(self, hp, save_dir, load_model):
        """
        
        Args:
            hp: HParams object
            save_dir: str
            load_model: str (directory with model and hparams)
        """
        self.hp = hp
        self.save_dir = save_dir
        self.load_model = load_model

        # TODO: should refactor StrokeToInstruction model
        hp = utils.load_hp(hp, load_model)
        model = StrokeToInstructionModel(hp, save_dir=None)  # save_dir=None means inference mode
        model.load_model(load_model)
        self.stroke2instruction = model

    def segment_all_data(self):
        for category in final_categories():
            for split in ['train']:
                print(category, split)
            # for split in ['train', 'valid', 'test']:
                ds = StrokeDataset(category, split)
                # TODO: make sure that StrokeDataset is properly normalized (main concern being that we used
                # the Stroke2Instruction model is trained on the ProgressionPair dataset, which uses the
                # stroke3 from ndjson and was already normalized / potentially normalized differently).
                loader = DataLoader(ds, batch_size=1, shuffle=False)
                for i, sample in enumerate(loader):
                    strokes, segmented = self.segment_sample(sample)  # [len, 5]
                    # data = {'sample': sample, 'segmented': segmented}
                    # out_fp = os.path.join(SEGMENTATIONS_PATH, split, '{}.pkl'.format(i))
                    out_fp = os.path.join(self.save_dir, category, split, '{}.json'.format(i))
                    utils.save_file(segmented, out_fp)

                    # Save original image too for comparisons
                    # TODO: this doesn't work properly because the conversion to ndjson seq isn't correct
                    strokes = strokes.cpu().numpy()
                    strokes[:, :2] *= 41.7  # unnormalize
                    ndjson_seq = convert_stroke5_to_ndjson_seq(strokes)
                    img = create_progression_image_from_ndjson_seq(ndjson_seq)
                    out_fp = os.path.join(self.save_dir, category, split, '{}.jpg'.format(i))
                    img.save(out_fp)

                    if i == 20:
                        break

    def construct_batch_of_segments_from_one_sample(self, strokes):
        """
        Args:
            strokes: [len, 5]
            

        Returns:
            batch: [n_pts (seq_len), n_segs, 5] FloatTensor
            n_penups: int
            seg_lens: list of ints, length n_segs
            seg_idx_map: dict
                
                {(0, 1): 0,
                 (0, 2): 1,
                 (0, 3): 2,
                 (0, 4): 3,
                 (0, 5): 4,
                 (1, 2): 5,
                 (1, 3): 6,
                 (1, 4): 7,
                 (1, 5): 8,
                 (2, 3): 9,
                 (2, 4): 10,
                 (2, 5): 11,
                 (3, 4): 12,
                 (3, 5): 13,
                 (4, 5): 14}
        """
        # get locations of segments using penup (4th point in stroke5 format)
        n_pts = strokes.size(0)
        strokes = strokes.cpu().numpy()
        pen_up = (np.where(strokes[:, 3] == 1)[0]).tolist()
        n_penups = len(pen_up)
        n_segs = int(n_penups * (n_penups + 1) / 2)

        # construct tensor of segments
        batch = np.zeros((n_segs, n_pts, 5))
        seg_lens = []
        seg_idx = 0
        seg_idx_map = {}  # maps tuple of (left_idx, right_idx) in terms of penups to seg_idx in batch
        pen_up = [0] + pen_up  # insert dummy
        for i in range(len(pen_up) - 1):  # i is left index
            for j in range(i+1, len(pen_up)):  # j is right index
                start_stroke_idx = pen_up[i]
                end_stroke_idx = pen_up[j]
                seg = strokes[start_stroke_idx:end_stroke_idx + 1]
                seg_len = len(seg)
                batch[seg_idx, :seg_len, :] = seg
                seg_lens.append(seg_len)
                seg_idx_map[(i,j)] = seg_idx
                seg_idx += 1

        batch = torch.Tensor(batch)
        batch = batch.transpose(0,1)  # [n_pts, n_segs, 5]
        batch = nn_utils.move_to_cuda(batch)

        return batch, n_penups, seg_lens, seg_idx_map

    def calculate_seg_probs(self, batch_of_segs, seg_lens, cats_idx):
        """
        Calculate the (log) probability of each segment
        (To be used as a error/goodness of fit for each segment)

        Args:
            batch_of_segs: [n_pts (seq_len), n_segs, 5]
            seg_lens: list of ints, length n_segs
            cats_idx: list of the same int, length n_segs

        Returns:
            final_probs [n_segs] array
        """
        probs, ids, texts = self.stroke2instruction.inference_pass(batch_of_segs, seg_lens, cats_idx)
        # probs: [n_segs, max_len, vocab]; texts: list of strs of length [n_segs]

        probs = probs.max(dim=-1)[0]  # [n_segs, max_len]; Using the max assumes greedy decoding basically
        final_probs = []

        # normalize by generated length
        n_segs = probs.size(0)
        for i in range(n_segs):
            eos_idx = (ids[i] == EOS_ID).nonzero()
            eos_idx = eos_idx.item() if (len(eos_idx) > 0) else probs.size(1)
            p = probs[i,:eos_idx + 1].log().sum() / float(eos_idx + 1)
            final_probs.append(p.item())
        final_probs = np.array(final_probs)  # [n_segs]

        return final_probs, texts

class SegmentationGreedyParsingModel(SegmentationModel):
    def __init__(self, hp, save_dir, load_model):
        super().__init__(hp, save_dir, load_model)

    def segment_sample(self, sample):
        """
        
        Args:
            batch_sample: sample from DataLoader of Strokedataset (batch_size=1)

        Returns:
            

        """
        strokes, stroke_lens, cats, cats_idx = sample
        strokes = strokes.transpose(0, 1).float()  # strokes: [len, 1, 5]
        strokes = nn_utils.move_to_cuda(strokes)
        strokes = strokes.squeeze(1)  # [len, 5]

        segs, n_penups, seg_lens, seg_idx_map = self.construct_batch_of_segments_from_one_sample(strokes)
        cats_idx = cats_idx.repeat(len(seg_lens))
        cats_idx = nn_utils.move_to_cuda(cats_idx)
        seg_probs, seg_texts = self.calculate_seg_probs(segs, seg_lens, cats_idx)

        from pprint import pprint

        # top level segmentation
        seg_idx = seg_idx_map[(0, n_penups)]
        segmented = [{'left': 0, 'right': n_penups, 'prob': seg_probs[seg_idx], 'text': seg_texts[seg_idx]}]
        segmented = self.split(0, n_penups, seg_idx_map, seg_probs, seg_texts, segmented)  # + 1see how seg_idx_map is calculated
        pprint(segmented)

        return strokes, segmented

    def split(self, left_idx, right_idx, seg_idx_map, seg_probs, seg_texts, segmented):
        # TODO: how best to store these?
        if (left_idx + 1) >= right_idx:
            return segmented
        else:
            max_prob = float('-inf')
            best_split_idx = None

            best_left_seg_text, best_right_seg_text = None, None
            best_left_seg_prob, best_right_seg_prob = None, None
            for split_idx in range(left_idx + 1, right_idx):
                left_seg_idx = seg_idx_map[(left_idx, split_idx)]
                right_seg_idx = seg_idx_map[(split_idx, right_idx)]
                left_seg_prob = seg_probs[left_seg_idx]
                right_seg_prob = seg_probs[right_seg_idx]
                prob = left_seg_prob + right_seg_prob

                if prob > max_prob:
                    best_left_seg_text, best_right_seg_text = seg_texts[left_seg_idx], seg_texts[right_seg_idx]
                    best_left_seg_prob, best_right_seg_prob = left_seg_prob, right_seg_prob
                    max_prob = prob
                    best_split_idx = split_idx

            segmented.append({'left': left_idx, 'right': best_split_idx,
                              'prob': best_left_seg_prob, 'text': best_left_seg_text})

            # TODO: call self.split on left side here so that it's prefix order?
            # That way it's like a stack
            segmented.append({'left': best_split_idx, 'right': right_idx,
                              'prob': best_right_seg_prob, 'text': best_right_seg_text})

            segmented = self.split(left_idx, best_split_idx, seg_idx_map, seg_probs, seg_texts, segmented)
            segmented = self.split(best_split_idx, right_idx, seg_idx_map, seg_probs, seg_texts, segmented)
            return segmented

class SegmentationBellmanModel(SegmentationModel):
    def __init__(self, hp, save_dir, load_model):
        super().__init__(hp, save_dir, load_model)

        self.model = None  # TODO: Load Model

    def segment_sample(self, sample):
        """
        
        Args:
            sample: 

        Returns:

        """
        # TODO: unpack sample?
        segs, n_segs = self.construct_batch_of_segments_from_one_sample(sample)
        costs = self.construct_costs(segs, n_segs)  # [n_segs, n_segs]

        # dynamic programming
        max_segs = self.hp.max_k if (self.hp.max_k) else n_segs
        Es = np.zeros(n_segs, max_segs + 1)  # TODO: off by 1 bs

        # Base case:
        # TODO do we have to fill in Es[:,0] ?
        Es[:,1] = costs[0,:]

        for i in range(n_segs):
            for k in range(1, max_segs + 1):
                min_cost = float('inf')
                for j in range(i):
                    cost = Es[j-1,k-1] + costs[j,i]  # TODO: out of bounds rn
                    if cost < min_cost:
                        min_cost = cost
                Es[i,k] = min_cost

        # TODO: backtrack
        # TODO: maybe refactor construct_costs to also return generated texts?
        # would need to generate text itself or modify one_forward_pass


        # TODO: save stroke segments, segment instructions, etc.
        utils.save_file(None, None)






if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='greedy_parsing')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = os.path.join(SEGMENTATIONS_PATH, opt.method)

    load_model = 'best_models/stroke2instruction/catsdecoder-dim_512-model_type_cnn_lstm-use_prestrokes_False/'
    hp.use_categories_enc = False
    hp.use_categories_dec = True  # backwards compatability (model was trained without that hparams)

    # load_model = 'runs/stroke2instruction/load_ae_556011f8/'

    if opt.method == 'bellman':
        model = SegmentationBellmanModel(hp, save_dir, load_model)
    elif opt.method == 'greedy_parsing':
        model = SegmentationGreedyParsingModel(hp, save_dir, load_model)

    model.segment_all_data()

    # Debugging saving of progressions from stroke data
    #
    # ds = StrokeDataset('cat', 'test')
    # # TODO: make sure that StrokeDataset is properly normalized (main concern being that we used
    # # the Stroke2Instruction model is trained on the ProgressionPair dataset, which uses the
    # # stroke3 from ndjson and was already normalized / potentially normalized differently).
    # loader = DataLoader(ds, batch_size=1, shuffle=False)
    # for i, sample in enumerate(loader):
    #     print(i)
    #     strokes, stroke_lens, cats, cats_idx = sample
    #     strokes = strokes.transpose(0, 1).float()  # strokes: [len, 1, 5]
    #     strokes = strokes.squeeze(1)  # [len, 5]
    #     strokes = strokes.cpu().numpy()
    #
    #     strokes[:,:2] *= 41.7  # unnormalize
    #
    #
    #     ndjson_seq = convert_stroke5_to_ndjson_seq(strokes)
    #
    #     # import pdb; pdb.set_trace()
    #     img = create_progression_image_from_ndjson_seq(ndjson_seq)
    #
    #     # save
    #     img.save(os.path.join('segs', '{}.jpg'.format(i)))
    #
    #     if i == 10:
    #         break
