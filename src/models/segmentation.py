# segmentation.py

"""
Currently uses trained Stroke2Instruction model to segment unseen sequences.

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python src/models/segmentation.py -ds progressionpair
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python src/models/segmentation.py -ds ndjson
"""

import argparse
import numpy as np
import os
from pprint import pprint
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import utils
from src.data_manager.quickdraw import QUICKDRAW_DATA_PATH, final_categories, \
    create_progression_image_from_ndjson_seq
from src.models.base.stroke_models import NdjsonStrokeDataset
from src.models.base.instruction_models import ProgressionPairDataset
from src.models.core import nn_utils
from src.models.instruction_gen import StrokeToInstructionModel, EOS_ID



sourSEGMENTATIONS_PATH = QUICKDRAW_DATA_PATH / 'segmentations'

##############################################################################
#
# Hyperparameters
#
##############################################################################
class HParams():
    def __init__(self):
        self.notes = ''

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

        # Load hp used to train model
        hp = utils.load_hp(hp, load_model)
        model = StrokeToInstructionModel(hp, save_dir=None)  # save_dir=None means inference mode
        model.load_model(load_model)
        self.stroke2instruction = model

    def segment_all_progressionpair_data(self):
        """
        Segment all samples in the ProgressionPairDataset
        """
        for split in ['train', 'valid', 'test']:
            print(split)
            ds = ProgressionPairDataset(split, return_full_stroke=True)
            loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ProgressionPairDataset.collate_fn)
            for i, sample in enumerate(loader):
                try:
                    id, category = loader.dataset.data[i]['id'], loader.dataset.data[i]['category']
                    out_dir = self.save_dir / split

                    # save segmentations
                    strokes, segmented = self.segment_sample(sample, dataset='progressionpair')
                    # TODO: save sample / strokes as well so that we have all the data in one place?
                    out_fp = out_dir / f'{category}_{id}.json'
                    utils.save_file(segmented, out_fp)

                    # save original image too for comparisons
                    ndjson_strokes = loader.dataset.data[i]['ndjson_strokes']
                    img = create_progression_image_from_ndjson_seq(ndjson_strokes)
                    out_fp = out_dir / f'{category}_{id}.jpg'
                    img.save(out_fp)

                except Exception as e:
                    print(e)
                    continue

    def segment_all_ndjson_data(self):
        """
        Segment all samples in the NdjsonStrokeDataset
        """
        for split in ['train', 'valid', 'test']:
            for category in final_categories():
                print(f'{split}: {category}')
                ds = NdjsonStrokeDataset(category, split)
                loader = DataLoader(ds, batch_size=1, shuffle=False)
                for i, sample in enumerate(loader):
                    try:
                        id, category = loader.dataset.data[i]['id'], loader.dataset.data[i]['category']
                        out_dir = self.save_dir / category
                        # note: we are NOT saving it into separate split categories in the case that
                        # we want to train on 30 categories and then do test on 5 held out categories.
                        # (i.e. keep it flexible to splitting within categories vs. across categories, which
                        # can be specified in that Dataset)
                        # TODO: should we do the same for ProgressionPair?

                        # save segmentations
                        strokes, segmented = self.segment_sample(sample, dataset='ndjson')
                        # TODO: save sample / strokes as well so that we have all the data in one place?
                        out_fp = out_dir / f'{id}.json'
                        utils.save_file(segmented, out_fp)

                        # save original image too for comparisons
                        ndjson_strokes = loader.dataset.data[i]['ndjson_strokes']
                        img = create_progression_image_from_ndjson_seq(ndjson_strokes)
                        out_fp = out_dir / f'{id}.jpg'
                        img.save(out_fp)

                    except Exception as e:
                        print(e)
                        continue

    def construct_batch_of_segments_from_one_sample(self, strokes):
        """
        Args:
            strokes: [len, 5] np array

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

        Returns: [n_segs] array
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

    def segment_sample(self, sample, dataset):
        """
        Args:
            sample: batch of samples from DataLoader of Strokedataset (batch_size=1)
            dataset: str

        Returns:
            strokes: TODO: why am I returning this?
            segmented: list of dicts
        """
        if dataset == 'ndjson':
            strokes, stroke_lens, cats, cats_idx = sample
        elif dataset == 'progressionpair':
            strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = sample

        strokes = strokes.transpose(0, 1).float()  # strokes: [len, 1, 5]
        strokes = nn_utils.move_to_cuda(strokes)
        strokes = strokes.squeeze(1)  # [len, 5]

        segs, n_penups, seg_lens, seg_idx_map = self.construct_batch_of_segments_from_one_sample(strokes)
        cats_idx = cats_idx.repeat(len(seg_lens))
        cats_idx = nn_utils.move_to_cuda(cats_idx)
        seg_probs, seg_texts = self.calculate_seg_probs(segs, seg_lens, cats_idx)

        # top level segmentation
        # initial instruction for entire sequence
        seg_idx = seg_idx_map[(0, n_penups)]
        segmented = [{'left': 0, 'right': n_penups, 'prob': seg_probs[seg_idx], 'text': seg_texts[seg_idx],
                      'id': uuid4().hex, 'parent': ''}]
        # recursively segment
        segmented = self.split(0, n_penups, seg_idx_map, seg_probs, seg_texts, segmented)  # + 1see how seg_idx_map is calculated
        # pprint(segmented)

        return strokes, segmented

    def split(self, left_idx, right_idx, seg_idx_map, seg_probs, seg_texts, segmented):
        """

        Args:
            left_idx: int
            right_idx: int
            seg_idx_map: dict (construct_batch_of_segments_from_one_sample())
            seg_probs: [n_segs] array
            seg_texts: [n_segs] strs
            segmented: list of dicts

        Returns: list of dicts
        """
        if (left_idx + 1) >= right_idx:
            return segmented

        # find best split
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

        # add left and right segment information
        # Note: append and splits must be called in the following order to get correct parent id
        parent_id = segmented[-1]['id']

        segmented.append({'left': left_idx, 'right': best_split_idx,
                          'prob': best_left_seg_prob, 'text': best_left_seg_text,
                          'id': uuid4().hex, 'parent': parent_id})
        segmented = self.split(left_idx, best_split_idx, seg_idx_map, seg_probs, seg_texts, segmented)
        segmented.append({'left': best_split_idx, 'right': right_idx,
                          'prob': best_right_seg_prob, 'text': best_right_seg_text,
                          'id': uuid4().hex, 'parent': parent_id})
        segmented = self.split(best_split_idx, right_idx, seg_idx_map, seg_probs, seg_texts, segmented)
        return segmented



if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='greedy_parsing')
    parser.add_argument('-ds', '--segment_dataset', default='progressionpair',
        help='Which dataset to segment -- "progressionpair" or "ndjson"')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    save_dir = SEGMENTATIONS_PATH / opt.method / opt.segment_dataset

    load_model = 'best_models/stroke2instruction/catsdecoder-dim_512-model_type_cnn_lstm-use_prestrokes_False/'
    hp.use_categories_enc = False
    hp.use_categories_dec = True  # backwards compatability (model was trained without that hparams)
    hp.unlikelihood_loss = False
    # TODO: find a better way to handle this...
    # TODO: we should probably save the hp and model path used to segment into save_dir

    if opt.method == 'greedy_parsing':
        model = SegmentationGreedyParsingModel(hp, save_dir, load_model)

    if opt.segment_dataset == 'progressionpair':
        model.segment_all_progressionpair_data()
    elif opt.segment_dataset == 'ndjson':
        model.segment_all_ndjson_data()
