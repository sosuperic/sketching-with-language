# segmentation.py

"""
Currently uses trained StrokesToInstruction model to segment unseen sequences.

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python src/models/segmentation.py -ds progressionpair --split_scorer instruction_to_strokes --save_subdir dec18
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python src/models/segmentation.py -ds ndjson
"""

import argparse
import copy
import numpy as np
import os
from pprint import pprint
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import utils
from src.data_manager.quickdraw import QUICKDRAW_DATA_PATH, final_categories, \
    create_progression_image_from_ndjson_seq, SEGMENTATIONS_PATH
from src.models.base.stroke_models import NdjsonStrokeDataset
from src.models.base.instruction_models import ProgressionPairDataset, LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH, map_sentence_to_index
from src.models.core import nn_utils
from src.models.instruction_to_strokes import InstructionToStrokesModel
from src.models.strokes_to_instruction import StrokesToInstructionModel, EOS_ID


##############################################################################
#
# Hyperparameters
#
##############################################################################
class HParams():
    def __init__(self):
        self.split_scorer = 'strokes_to_instruction'  # 'instruction_to_strokes'

        # self.strokes_to_instruction_dir = 'best_models/strokes_to_instruction/catsdecoder-dim_512-model_type_cnn_lstm-use_prestrokes_False/'
        self.strokes_to_instruction_dir = 'runs/strokes_to_instruction/bigsweep/condition_on_hc_True-dim_256-dropout_0.2-lr_0.0005-model_type_lstm-n_dec_layers_4-n_enc_layers_4-use_categories_dec_True-use_categories_enc_False-use_prestrokes_False'
        self.instruction_to_strokes_dir = 'runs/instruction_to_strokes/dec17/cond_instructions_initdec-dec_dim_512-enc_dim_512-lr_0.001-model_type_decodergmm/'

        self.notes = ''

##############################################################################
#
# Model
#
##############################################################################

class SegmentationModel(object):
    def __init__(self, hp, save_dir):
        """
        Args:
            hp: HParams object
            save_dir: str
        """
        self.hp = hp
        self.save_dir = save_dir

        # Load hp used to train model

        self.s2i_hp = utils.load_hp(copy.deepcopy(hp), hp.strokes_to_instruction_dir)
        self.strokes_to_instruction = StrokesToInstructionModel(self.s2i_hp, save_dir=None)  # save_dir=None means inference mode
        self.strokes_to_instruction.load_model(hp.strokes_to_instruction_dir)
        self.strokes_to_instruction.cuda()

        if hp.split_scorer == 'instruction_to_strokes':
            self.i2s_hp = utils.load_hp(copy.deepcopy(hp), hp.instruction_to_strokes_dir)
            self.instruction_to_strokes = InstructionToStrokesModel(self.i2s_hp, save_dir=None)
            self.instruction_to_strokes.load_model(hp.instruction_to_strokes_dir)  # TODO: change param for load_model
            self.instruction_to_strokes.cuda()


        # TODO: this should be probably be contained in some model...
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)

    def segment_all_progressionpair_data(self):
        """
        Segment all samples in the ProgressionPairDataset
        """
        for split in ['train', 'valid', 'test']:
            print(split)
            ds = ProgressionPairDataset(split, use_full_drawings=True)
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

    def calculate_seg_scores(self, batch_of_segs, seg_lens, cats_idx):
        """
        Calculate
        Calculate the (log) probability of each segment
        (To be used as a error/goodness of fit for each segment)

        Args:
            batch_of_segs: [n_pts (seq_len), n_segs, 5] CudaFloatTensor
            seg_lens: list of ints, length n_segs
            cats_idx: list of the same int, length n_segs

        Returns:
            scores ([n_segs] np array)
            texts (list): n_segs list of strings
        """
        if self.hp.split_scorer == 'strokes_to_instruction':
            with torch.no_grad():
                probs, ids, texts = self.strokes_to_instruction.inference_pass(batch_of_segs, seg_lens, cats_idx)
            probs = probs.max(dim=-1)[0]  # [n_segs, max_len]; Using the max assumes greedy decoding basically

            # normalize by generated length
            final_probs = []
            n_segs = probs.size(0)
            for i in range(n_segs):
                eos_idx = (ids[i] == EOS_ID).nonzero()
                eos_idx = eos_idx.item() if (len(eos_idx) > 0) else probs.size(1)
                p = probs[i,:eos_idx + 1].log().sum() / float(eos_idx + 1)
                final_probs.append(p.item())
            scores = np.array(final_probs)  # [n_segs]
            return scores, texts

        elif self.hp.split_scorer == 'instruction_to_strokes':
            with torch.no_grad():
                probs, ids, texts = self.strokes_to_instruction.inference_pass(batch_of_segs, seg_lens, cats_idx)
            text_indices_list = [map_sentence_to_index(text, self.token2idx) for text in texts]

            # Construct inputs to instruction_to_strokes model
            bsz = batch_of_segs.size(1)
            text_lens = [len(t) for t in text_indices_list]
            max_len = max(text_lens)
            text_indices = np.zeros((max_len, bsz))
            for i, indices in enumerate(text_indices_list):
                text_indices[:len(indices), i] = indices
            text_indices = nn_utils.move_to_cuda(torch.LongTensor(text_indices))

            cats = ['' for _ in range(bsz)]  # dummy
            urls = ['' for _ in range(bsz)]  # dummy
            batch = (batch_of_segs, seg_lens, texts, text_lens, text_indices, cats, cats_idx, urls)

            with torch.no_grad():
                result = self.instruction_to_strokes.one_forward_pass(batch, average_loss=False)  #
                scores = result['loss'].cpu().numpy().astype(np.float64)  # float32 doesn't serialize to json for some reason

            return scores, texts

class SegmentationGreedyParsingModel(SegmentationModel):
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

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
        seg_scores, seg_texts = self.calculate_seg_scores(segs, seg_lens, cats_idx)

        # top level segmentation
        # initial instruction for entire sequence
        seg_idx = seg_idx_map[(0, n_penups)]
        segmented = [{'left': 0, 'right': n_penups, 'score': seg_scores[seg_idx], 'text': seg_texts[seg_idx],
                      'id': uuid4().hex, 'parent': ''}]
        # recursively segment
        segmented = self.split(0, n_penups, seg_idx_map, seg_scores, seg_texts, segmented)
        # pprint(segmented)

        return strokes, segmented

    def split(self, left_idx, right_idx, seg_idx_map, seg_scores, seg_texts, segmented):
        """

        Args:
            left_idx: int
            right_idx: int
            seg_idx_map: dict (construct_batch_of_segments_from_one_sample())
            seg_scores: [n_segs] array
            seg_texts: [n_segs] strs
            segmented: list of dicts

        Returns: list of dicts
        """
        if (left_idx + 1) >= right_idx:
            return segmented

        # find best split
        max_score = float('-inf')
        best_split_idx = None
        best_left_seg_text, best_right_seg_text = None, None
        best_left_seg_score, best_right_seg_score = None, None
        for split_idx in range(left_idx + 1, right_idx):
            left_seg_idx = seg_idx_map[(left_idx, split_idx)]
            right_seg_idx = seg_idx_map[(split_idx, right_idx)]
            left_seg_score = seg_scores[left_seg_idx]
            right_seg_score = seg_scores[right_seg_idx]
            score = left_seg_score + right_seg_score

            if score > max_score:
                best_left_seg_text, best_right_seg_text = seg_texts[left_seg_idx], seg_texts[right_seg_idx]
                best_left_seg_score, best_right_seg_score = left_seg_score, right_seg_score
                max_score = score
                best_split_idx = split_idx

        # add left and right segment information
        # Note: append and splits must be called in the following order to get correct parent id
        parent_id = segmented[-1]['id']

        segmented.append({'left': left_idx, 'right': best_split_idx,
                          'score': best_left_seg_score, 'text': best_left_seg_text,
                          'id': uuid4().hex, 'parent': parent_id})
        segmented = self.split(left_idx, best_split_idx, seg_idx_map, seg_scores, seg_texts, segmented)
        segmented.append({'left': best_split_idx, 'right': right_idx,
                          'score': best_right_seg_score, 'text': best_right_seg_text,
                          'id': uuid4().hex, 'parent': parent_id})
        segmented = self.split(best_split_idx, right_idx, seg_idx_map, seg_scores, seg_texts, segmented)
        return segmented



if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    parser.add_argument('--save_subdir')
    # Model
    parser.add_argument('--method', default='greedy_parsing')
    parser.add_argument('-ds', '--segment_dataset', default='progressionpair',
                        help='Which dataset to segment -- "progressionpair" or "ndjson"')
    opt = parser.parse_args()

    # Setup
    nn_utils.setup_seeds()
    save_dir = SEGMENTATIONS_PATH / opt.method / opt.segment_dataset / datetime.today().strftime('%b%d_%Y') / hp.split_scorer / opt.save_subdir
    # TODO: find a better way to handle this...
    hp.use_categories_enc = False
    hp.use_categories_dec = True  # backwards compatability (InstructionToStrokes model was trained without that hparams)
    hp.unlikelihood_loss = False
    # TODO: we should probably 1) set decoding hparams (e.g. greedy, etc.), 2) save the hp
    # utils.save_file(vars(hp), save_dir / 'hp.json')
    utils.save_run_data(save_dir, hp)

    # Init model and segment
    if opt.method == 'greedy_parsing':
        model = SegmentationGreedyParsingModel(hp, save_dir)
    if opt.segment_dataset == 'progressionpair':
        model.segment_all_progressionpair_data()
    elif opt.segment_dataset == 'ndjson':
        model.segment_all_ndjson_data()
