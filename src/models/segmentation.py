# segmentation.py

"""
Model for training / doing DP segmentation method
"""

import argparse
import numpy as np
import os

import torch

from src.data_manager.quickdraw import QUICKDRAW_DATA_PATH
from src.models import nn_utils
from src import utils

from src.models.sketch_rnn import SketchDataset
from src.data_manager.quickdraw import final_categories


from torch.utils.data import Dataset, DataLoader
# TODO: import StrokeDataset

SEGMENTATIONS_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'segmentations')

##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        self.max_k = None  # for DP method; int or None
        self.notes = ''

##############################################################################
#
# Model
#
##############################################################################

class Segmentation(object):
    def __init__(self, hp, save_dir=None):
        self.hp = hp
        self.save_dir = save_dir

    def segment_all_data(self):
        for category in final_categories():
            for split in ['train', 'valid', 'test']:
                ds = SketchDataset(split, category=category)
                loader = DataLoader(ds, batch_size=1, shuffle=False)
                for segmented in self.segment(loader):
                    pass # TODO: save segmented

    def segment(self, data_loader):
        for sample in data_loader:
            yield self.segment_sample(sample)

    def segment_sample(self, sample):
        """
        Args:
            sample: one sample from  

        Returns: dict? TODO
        """
        pass

class SegmentationDP(Segmentation):
    def __init__(self, hp, load_model=None, save_dir=None):
        super().__init__(hp, save_dir)

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

    def construct_costs(self, batch_of_segments, n_segs):
        """
        Make cost matrix to be used in DP

        Args:
            batch_of_segments: [n_segs^2, sample_len] (n_segs = number of pen_ups)
            n_segs: int

        Returns: [N, N] numpy array
        """
        costs = -torch.log(self.model.one_forward_pass(batch_of_segments)['normed_probs'])  # TODO: add this key in instruction_gen   # [n_segs ^ 2] LongTensor?
        # TODO: do -torch.log here?
        costs = costs.view(n_segs, n_segs)
        costs = costs.cpu().numpy()
        return costs


    def construct_batch_of_segments_from_one_sample(self, strokes):
        """
        Args:
            strokes: [len, 3 or 5] np array (either stroke3 or stroke5 format)

        Returns: [N^2, sample_len]
            N^2 where N is number of penups 
            sample_len (max length of a segment is sample len)
        """
        n_pts = strokes.size(0)

        pen_up = np.where(strokes[:, 2] == 1)[0].tolist()
        n_segs = len(pen_up)  # TODO: look into off by one, penup vs pendown nonsense

        batch = np.zeros(n_segs, n_pts)
        cur = 0
        for i in range(n_segs-1):
            for j in range(i+1, n_segs):
                start_stroke_idx = pen_up[i]
                end_stroke_idx = pen_up[j]
                seg = strokes[start_stroke_idx:end_stroke_idx]  # +1 somewhere? TODO
                batch[cur,:len(seg)] = seg

        batch = torch.Tensor(batch)

        return batch, n_segs





if __name__ == '__main__':

    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=None)
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    if opt.method == 'dp':
        # TODO: should dataset be part of directory path name? Where is data loaded?
        save_dir = os.path.join(SEGMENTATIONS_PATH, 'dp', run_name)
        model = SegmentationDP(hp, save_dir)
        utils.save_run_data(save_dir, hp)
        model = SegmentationDP(hp)
        model.segment_all_data()
