# segmentation.py

"""
Currently uses trained StrokesToInstruction model to segment unseen sequences.

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python src/models/segmentation.py -ds progressionpair
"""

import argparse
import copy
from datetime import datetime
import numpy as np
from PIL import Image
import os
from pprint import pprint
from uuid import uuid4

import spacy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import SEGMENTATIONS_PATH, LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH, \
    BEST_STROKES_TO_INSTRUCTION_PATH, BEST_INSTRUCTION_TO_STROKES_PATH
from src import utils
from src.data_manager.quickdraw import final_categories, create_progression_image_from_ndjson_seq
from src.models.base.stroke_models import NdjsonStrokeDataset
from src.models.base.instruction_models import ProgressionPairDataset, map_sentence_to_index, \
    DrawingsAsImagesAnnotatedDataset
from src.models.core import experiments, nn_utils
from src.models.instruction_to_strokes import InstructionToStrokesModel
from src.models.strokes_to_instruction import HParams as s2i_default_hparams
from src.models.strokes_to_instruction import StrokesToInstructionModel, EOS_ID


##############################################################################
#
# Hyperparameters
#
##############################################################################
class HParams():
    def __init__(self):
        self.split_scorer = 'strokes_to_instruction'  # 'instruction_to_strokes'

        self.score_parent_child_text_sim = False  # similarity b/n parent text and children text (concatenated)
        self.score_exponentiate = 1.0  # seg1_score ** alpha * seg2_score ** alpha
        self.score_childinst_parstroke = False   # P(parent_strokes | [child_inst1, child_inst2])

        self.strokes_to_instruction_dir = BEST_STROKES_TO_INSTRUCTION_PATH
        self.instruction_to_strokes_dir = BEST_INSTRUCTION_TO_STROKES_PATH
        self.notes = ''

        # Dataset (for larger ndjson dataset)
        self.categories = 'all'
        self.max_per_category = 2750

##############################################################################
#
# Utils
#
##############################################################################

def remove_stopwords(nlp, text):
    """
    Args:
        nlp (spacy  model): [description]
        text (str):

    Returns:
        str
    """
    doc = nlp(text.lower())
    result = [token.text for token in doc if token.text not in nlp.Defaults.stop_words]
    result = ' '.join(result)
    return result


def prune_seg_tree(seg_tree, prob_threshold=0):
    """
    Args:
        seg_tree (list of dicts):
            In order of splits as done by SegmentationModel.
                E.g. 0-4, then 0-3, then 0-1, then 1-3, then 1-2, then 2-3, then 3-4

            Each dict contains data about that segment.
                'left': start idx
                'right': end idx
                'id':
                'parent': parent's id
                'text':
                'score': Currently P(I|S) for that segment

        prob_threshold (float): score must be greater than prob_threshold

    Returns seg_tree (list of dicts)
        all segments that fall below prob_threshold removed (including each segment's subsegments)
    """
    pruned = [seg_tree[0]]  # must have root
    added_ids = set([seg_tree[0]['id']])
    for i in range(1, len(seg_tree)):
        seg = seg_tree[i]
        if seg['score'] > prob_threshold:
            if seg['parent'] in added_ids:  # parent must have been added (i.e. above threshold)
                pruned.append(seg)
                added_ids.add(seg['id'])
    return pruned

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
        self.s2i_hp = experiments.load_hp(copy.deepcopy(hp), hp.strokes_to_instruction_dir)
        default_s2i_hp = s2i_default_hparams()
        # For backwards compatibility:
        # hparams may have been added since model was trained; add them to s2i_hp
        for k, v in vars(default_s2i_hp).items():
            if not hasattr(self.s2i_hp, k):
                setattr(self.s2i_hp, k, v)
        self.s2i_hp.drawing_type = 'stroke'  # TODO: this should be image if we switch to the images model

        self.strokes_to_instruction = StrokesToInstructionModel(self.s2i_hp, save_dir=None)  # save_dir=None means inference mode
        self.strokes_to_instruction.load_model(hp.strokes_to_instruction_dir)
        self.strokes_to_instruction.cuda()

        if (hp.split_scorer == 'instruction_to_strokes') or (hp.score_childinst_parstroke):
            self.i2s_hp = experiments.load_hp(copy.deepcopy(hp), hp.instruction_to_strokes_dir)
            # TODO: should do same backwards compatibility as above
            self.instruction_to_strokes = InstructionToStrokesModel(self.i2s_hp, save_dir=None)
            self.instruction_to_strokes.load_model(hp.instruction_to_strokes_dir)  # TODO: change param for load_model
            self.instruction_to_strokes.cuda()

        if hp.score_parent_child_text_sim:
            spacy.prefer_gpu()
            self.nlp = spacy.load('en_core_web_md')

        # TODO: this should be probably be contained in some model...
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)

    def segment_all_progressionpair_data(self):
        """
        Segment all samples in the ProgressionPairDataset
        """
        for split in ['train', 'valid', 'test']:
            print(split)
            if self.s2i_hp.drawing_type == 'stroke':
                self.ds = ProgressionPairDataset(split, use_full_drawings=True)
                loader = DataLoader(self.ds, batch_size=1, shuffle=False, collate_fn=ProgressionPairDataset.collate_fn)
            elif self.s2i_hp.drawing_type == 'image':
                self.ds = DrawingsAsImagesAnnotatedDataset(split, images=self.s2i_hp.images, data_aug_on_text=False)
                loader = DataLoader(self.ds, batch_size=1, shuffle=False, collate_fn=DrawingsAsImagesAnnotatedDataset.collate_fn)

            for i, sample in enumerate(loader):
                try:
                    id, category = loader.dataset.data[i]['id'], loader.dataset.data[i]['category']
                    out_dir = self.save_dir / split

                    if self.s2i_hp.drawing_type == 'image':
                        sample = loader.dataset.data[i]  # contains the fp, n_segments data we need

                    # save segmentations
                    segmented = self.segment_sample(sample, dataset='progressionpair')
                    # TODO: save sample / strokes as well so that we have all the data in one place?
                    out_fp = out_dir / f'{category}_{id}.json'
                    utils.save_file(segmented, out_fp)

                    # save original image too for comparisons
                    # TODO: image dataset doesn't have ndjson_strokes
                    # ndjson_strokes = loader.dataset.data[i]['ndjson_strokes']
                    # img = create_progression_image_from_ndjson_seq(ndjson_strokes)
                    out_fp = out_dir / f'{category}_{id}.jpg'
                    open(out_fp, 'a').close()
                    # img.save(out_fp)

                except Exception as e:
                    print(e)
                    continue

    def segment_all_ndjson_data(self):
        """
        Segment all samples in the NdjsonStrokeDataset
        """
        for split in ['train', 'valid', 'test']:
            for category in final_categories():
                # Skip if not in hparam's categories list
                if (self.hp.categories != 'all') and (category not in self.hp.categories):
                    continue
                print(f'{split}: {category}')
                # ds = NdjsonStrokeDataset(category, split)
                ds = NdjsonStrokeDataset(category, split, max_per_category=3000)
                loader = DataLoader(ds, batch_size=1, shuffle=False)
                n_segd = 0
                for i, sample in enumerate(loader):
                    try:
                        id, category = loader.dataset.data[i]['id'], loader.dataset.data[i]['category']
                        out_dir = self.save_dir / category
                        out_fp = out_dir / f'{id}.json'
                        if os.path.exists(out_fp):
                            continue
                        # note: we are NOT saving it into separate split categories in the case that
                        # we want to train on 30 categories and then do test on 5 held out categories.
                        # (i.e. keep it flexible to splitting within categories vs. across categories, which
                        # can be specified in that Dataset)
                        # TODO: should we do the same for ProgressionPair?

                        # save segmentations
                        segmented = self.segment_sample(sample, dataset='ndjson')
                        # TODO: save sample / strokes as well so that we have all the data in one place?
                        utils.save_file(segmented, out_fp)

                        # save original image too for comparisons
                        ndjson_strokes = loader.dataset.data[i]['ndjson_strokes']
                        img = create_progression_image_from_ndjson_seq(ndjson_strokes)
                        out_fp = out_dir / f'{id}.jpg'
                        img.save(out_fp)

                        n_segd += 1
                        if n_segd == self.hp.max_per_category:
                            break

                    except Exception as e:
                        print(e)
                        continue

    def construct_batch_of_segments_from_one_sample_image(self, sample):
        """
        See construct_batch_of_segments_from_one_sample_stroke for more details

        Args:
            sample (dict): one data point from DrawingAsImage...Dataset
                contains fp's and n_segments
        """
        fn = os.path.basename(sample['post_seg_fp'])  # data/quickdraw/precurrentpost/data/pig/5598031527280640/7-10.jpg
        start, end = fn.strip('.jpg').split('-')
        end = int(end)
        n_penups = end

        seg_idx = 0
        seg_idx_map = {}  # maps tuple of (left_idx, right_idx) in terms of penups to seg_idx in batch
        batch = []
        for i in range(n_penups):  # i is left index
            for j in range(i+1, n_penups + 1):  # j is right index
                img = self.ds._construct_rank_image(i, j, n_penups, sample)
                batch.append(img)
                seg_idx_map[(i,j)] = seg_idx
                seg_idx += 1

        seg_lens = [1 for _ in range(len(batch))]  # dummy lengths (not used)

        batch = np.stack(batch)  # [n_segs, C, H, W]
        batch = torch.Tensor(batch)
        batch = batch.transpose(0,1)  # [C, n_segs, H, W]
        batch = nn_utils.move_to_cuda(batch)

        return batch, n_penups, seg_lens, seg_idx_map

    def construct_batch_of_segments_from_one_sample_stroke(self, strokes):
        """
        Args:
            strokes: [len, 5] np array

        Returns:
            batch: [n_pts (seq_len), n_segs, 5] FloatTensor
            n_penups: int
            seg_lens: list of ints, length n_segs
            seg_idx_map: dict
                Maps penup_idx tuples to seg_idx
                Example with 5 penups
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

    def _calc_instruction_to_strokes_score(self, batch_of_segs, seg_lens, texts, cats_idx):
        """
        P(S|I). Note that it's the prob, not the loss (NLL) returned by the model.

        Args:
            batch_of_segs: [n_pts (seq_len), n_segs, 5] CudaFloatTensor
            seg_lens: list of ints, length n_segs
            texts (list): n_segs list of strings
            cats_idx: list of the same int, length n_segs

        Returns:
            scores: (n_segs) np array
        """
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
            result = self.instruction_to_strokes.one_forward_pass(batch, average_loss=False)  # [n_segs]?
            scores = result['loss'].cpu().numpy().astype(np.float64)  # float32 doesn't serialize to json for some reason
            scores = np.exp(-scores)  # map losses (NLL) to probs

        return scores

    def calculate_seg_scores(self, batch_of_segs, seg_lens, cats_idx, seg_idx_map):
        """
        Calculate
        Calculate the (log) probability of each segment
        (To be used as a error/goodness of fit for each segment)

        Args:
            batch_of_segs: [n_pts (seq_len), n_segs, 5] CudaFloatTensor  (n_segs is the "batch")
            seg_lens: list of ints, length n_segs
            cats_idx: list of the same int, length n_segs
            seg_idx_map: dict
                Maps penup_idx tuples to seg_idx

        Returns:
            scores ([n_segs] np array)
            texts (list): n_segs list of strings
            parchild_scores: [n_par_segs] np arrray, indexed by paridx; n_par_segs != n_segs
            leftrightsegidx_to_paridx: tuple (left_seg_id, right_seg_idx) to int
                paridx indexes into parchild_scores
                left_seg_idx, right_seg_idx index into batch_of_segs and seg_lens
                (note: seg_idx_map maps pen_up index into seg_idx)
        """
        with torch.no_grad():
            probs, ids, texts = self.strokes_to_instruction.inference_pass(batch_of_segs, seg_lens, cats_idx)
            # probs: [n_segs, max_len, vocab]

        if self.hp.split_scorer == 'strokes_to_instruction':
            # normalize by generated length
            probs = probs.max(dim=-1)[0]  # [n_segs, max_len]; Using the max assumes greedy decoding basically
            final_probs = []
            n_segs = probs.size(0)
            for i in range(n_segs):
                eos_idx = (ids[i] == EOS_ID).nonzero()
                eos_idx = eos_idx.item() if (len(eos_idx) > 0) else probs.size(1)
                p = probs[i,:eos_idx + 1].sum() / float(eos_idx + 1)
                final_probs.append(p.item())
            scores = np.array(final_probs)  # [n_segs]

        elif self.hp.split_scorer == 'instruction_to_strokes':
            scores = self._calc_instruction_to_strokes_score(batch_of_segs, seg_lens, texts, cats_idx)  # probs


        # Calculate score based on P(parent_strokes | [child_inst1, child_inst2]). This is done with a
        # trained I2S model, with the parent segment as the target, and the concat'd instructions
        # as input
        #
        # Plan: construct a batch of parent_segs.
        # - Use penup indices to map into seg_indices (seg_idx indexes into batch_of_segs, texts, etc.)
        # - Use those seg_indices to get the child texts
        # - Also get the parent segment
        # - Update leftrightsegidx_to_parchildidx, which we need to get the score later in split()
        # - Compute scores for this batch, where batch_size is n_parent_segs
        parchild_scores = None
        leftrightsegidx_to_parchildidx = {}
        if self.hp.score_childinst_parstroke:

            parent_segs = []
            parent_seg_lens = []
            child_texts = []

            n_segs = len(seg_lens)
            left_penups, right_penups = zip(*seg_idx_map.keys())
            n_penups = max(right_penups) + 1
            parchild_idx = 0  # n_parent_segs is different from n_segs)

            # Iterate over penups (and not seg_indices, i.e. len(texts)) because we want to
            # choose segments that are next to each other (and hence share a parent).
            # This is similar to the way that seg_idx_map was created (left_penup, right_penup) -> seg_idx
            for left_penup in range(n_penups - 2):
                for middle_penup in range(left_penup + 1, n_penups -1):
                    for right_penup in range(middle_penup + 1, n_penups):

                        # Get the corresponding seg_idxs, which allows us to get the text
                        left_seg_idx = seg_idx_map[(left_penup, middle_penup)]
                        right_seg_idx = seg_idx_map[(middle_penup, right_penup)]
                        left_text = texts[left_seg_idx]
                        right_text = texts[right_seg_idx]
                        left_right_text = ' '.join([left_text, right_text])  # concat child texts
                        child_texts.append(left_right_text)

                        # get the parent_segment
                        parent_seg_idx = seg_idx_map[(left_penup, right_penup)]
                        parent_seg, parent_seg_len = batch_of_segs[:,parent_seg_idx,:], seg_lens[parent_seg_idx]
                        parent_segs.append(parent_seg)
                        parent_seg_lens.append(parent_seg_len)

                        # update mapping. Later, when we are splitting, we can use this dict to get the
                        # childinst_parstroke score
                        # TODO: I guess I could've mapped penups to parchild_idx, so that both indices dictionaries
                        # (this one and seg_idx_map) have penups as keys...
                        leftrightsegidx_to_parchildidx[(left_seg_idx, right_seg_idx)] = parchild_idx
                        parchild_idx += 1

            # Compute scores
            parent_segs = torch.stack(parent_segs, dim=1)  # [max_len, n_parent_segs, 5]
            parchild_scores = self._calc_instruction_to_strokes_score(parent_segs, parent_seg_lens, child_texts, cats_idx)  # [n_parent_segs]

        return scores, texts, parchild_scores, leftrightsegidx_to_parchildidx

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

        Note:
            There are several different indices / mappings.
            1) penups (i.e. number of penups). One penup = one segment.
            2) seg_idx. Indexes into batch_of_segs.
                seg_idx_map: (left_penup, right_penup) -> seg_idx
                    e.g. (0,1) -> 0, (0,2) -> 1, (0,3) -> 2, (1,2) -> 3, (1,3) -> 4, (2,3) -> 5
            3) parchild_idx. Indexes into parchild_scores, i.e. P(S | [I1, I2]).
                leftrightsegidx_to_parchildidx: (left_seg_idx, right_seg_idx) -> par_child_idx

        """

        if self.s2i_hp.drawing_type == 'stroke':
            if dataset == 'ndjson':
                strokes, stroke_lens, cats, cats_idx = sample
            elif dataset == 'progressionpair':
                strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = sample

            strokes = strokes.transpose(0, 1).float()  # strokes: [len, 1, 5]
            strokes = nn_utils.move_to_cuda(strokes)
            strokes = strokes.squeeze(1)  # [len, 5]

            segs, n_penups, seg_lens, seg_idx_map = self.construct_batch_of_segments_from_one_sample_stroke(strokes)
            cats_idx = cats_idx.repeat(len(seg_lens))
            cats_idx = nn_utils.move_to_cuda(cats_idx)

        elif self.s2i_hp.drawing_type == 'image':
            if dataset == 'ndjson':
                raise NotImplementedError
            elif dataset == 'progressionpair':
                segs, n_penups, seg_lens, seg_idx_map = self.construct_batch_of_segments_from_one_sample_image(sample)
                cats_idx = self.ds.cat2idx[sample['category']]
                cats_idx = torch.LongTensor([cats_idx for _ in range(len(seg_lens))])
                cats_idx = nn_utils.move_to_cuda(cats_idx)

        seg_scores, seg_texts, parchild_scores, leftrightsegidx_to_parchildidx = self.calculate_seg_scores(segs, seg_lens, cats_idx, seg_idx_map)

        # top level segmentation
        # initial instruction for entire sequence
        seg_idx = seg_idx_map[(0, n_penups)]
        segmented = [{'left': 0, 'right': n_penups, 'score': seg_scores[seg_idx], 'text': seg_texts[seg_idx],
                      'id': uuid4().hex, 'parent': ''}]
        # recursively segment
        segmented = self.split(0, n_penups, seg_idx_map, seg_scores, seg_texts, segmented,
                               parchild_scores, leftrightsegidx_to_parchildidx)

        return segmented

    def split(self, left_penup_idx, right_penup_idx, seg_idx_map, seg_scores, seg_texts, segmented,
              parchild_scores, leftrightsegidx_to_parchildidx):
        """

        Args:
            left_penup_idx: int (idx of penups)
            right_penup_idx: int  (idx of penups)
            seg_idx_map: dict (construct_batch_of_segments_from_one_sample())
                maps from (left_penup_idx, right_penup_idx) to seg_idx
                seg_idx is index within batch of segments
            seg_scores: [n_segs] array
            seg_texts: [n_segs] strs
            segmented: list of dicts
            parchild_scores: indexed by paridx, which is obtained by leftrightsegidx_to_paridx
            leftrightsegidx_to_parchildidx: maps from segidx to paridx
                parchildidx is index within parchild_scores

        Returns: list of dicts
        """
        if (left_penup_idx + 1) >= right_penup_idx:
            return segmented

        # find best split
        max_score = float('-inf')
        best_split_penup_idx = None
        best_left_seg_text, best_right_seg_text = None, None
        best_left_seg_score, best_right_seg_score = None, None
        for split_penup_idx in range(left_penup_idx + 1, right_penup_idx):
            left_seg_idx = seg_idx_map[(left_penup_idx, split_penup_idx)]
            right_seg_idx = seg_idx_map[(split_penup_idx, right_penup_idx)]
            left_seg_score = seg_scores[left_seg_idx]
            right_seg_score = seg_scores[right_seg_idx]
            score = left_seg_score ** self.hp.score_exponentiate * right_seg_score ** self.hp.score_exponentiate

            # P(S | [I1, I2])
            if self.hp.score_childinst_parstroke:
                childinst_parseg_idx = leftrightsegidx_to_parchildidx[(left_seg_idx, right_seg_idx)]
                childinst_parseg_score = parchild_scores[childinst_parseg_idx]

            # compute similarity between concatenated children instructions and parent instruction
            # (i.e. instruction for entire parent segment)
            if self.hp.score_parent_child_text_sim:
                # Get parent segment and texts
                parent_seg_idx = seg_idx_map[(left_penup_idx, right_penup_idx)]
                parent_seg_text = remove_stopwords(self.nlp, seg_texts[parent_seg_idx])
                left_seg_text = remove_stopwords(self.nlp, seg_texts[left_seg_idx])
                right_seg_text = remove_stopwords(self.nlp, seg_texts[right_seg_idx])

                left_right_text = left_seg_text + ' ' + right_seg_text
                parent_children_sim_score = self.nlp(parent_seg_text).similarity(self.nlp(left_right_text))
                score += parent_children_sim_score

            if score > max_score:
                best_left_seg_text, best_right_seg_text = seg_texts[left_seg_idx], seg_texts[right_seg_idx]
                best_left_seg_score, best_right_seg_score = left_seg_score, right_seg_score
                max_score = score
                best_split_penup_idx = split_penup_idx

        # add left and right segment information
        # Note: append and splits must be called in the following order to get correct parent id
        parent_id = segmented[-1]['id']

        # add left
        segmented.append({'left': left_penup_idx, 'right': best_split_penup_idx,
                          'score': best_left_seg_score, 'text': best_left_seg_text,
                          'id': uuid4().hex, 'parent': parent_id})
        # recursively split left
        segmented = self.split(left_penup_idx, best_split_penup_idx, seg_idx_map, seg_scores, seg_texts, segmented,
                            parchild_scores, leftrightsegidx_to_parchildidx)

        # add right
        segmented.append({'left': best_split_penup_idx, 'right': right_penup_idx,
                          'score': best_right_seg_score, 'text': best_right_seg_text,
                          'id': uuid4().hex, 'parent': parent_id})
        # recursively split right
        segmented = self.split(best_split_penup_idx, right_penup_idx, seg_idx_map, seg_scores, seg_texts, segmented,
                            parchild_scores, leftrightsegidx_to_parchildidx)
        return segmented






if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--method', default='greedy_parsing')
    parser.add_argument('--groupname', default=None)
    parser.add_argument('-ds', '--segment_dataset', default='progressionpair',
                        help='Which dataset to segment -- "progressionpair" or "ndjson"')
    opt = parser.parse_args()

    # Setup
    nn_utils.setup_seeds()
    save_dir = SEGMENTATIONS_PATH / opt.method / opt.segment_dataset / datetime.today().strftime('%b%d_%Y') / hp.split_scorer
    if opt.groupname is not None:
        save_dir = save_dir / opt.groupname

    # TODO: we should probably 1) set decoding hparams (e.g. greedy, etc.), 2) save the hp
    # utils.save_file(vars(hp), save_dir / 'hp.json')
    experiments.save_run_data(save_dir, hp,  ask_if_exists=False)
    # ask_if_exists=False because sweep calls it with different categories, but all saved to same directory

    # Init model and segment
    if opt.method == 'greedy_parsing':
        model = SegmentationGreedyParsingModel(hp, save_dir)
    if opt.segment_dataset == 'progressionpair':
        model.segment_all_progressionpair_data()
    elif opt.segment_dataset == 'ndjson':
        model.segment_all_ndjson_data()
