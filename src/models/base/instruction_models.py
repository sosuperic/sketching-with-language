# instruction_models.py

"""
Instruction (annotations from MTurk) related models and dataset
"""

import os
import random

import numpy as np
import nltk
import PIL
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision

import haste_pytorch as haste

from config import LABELED_PROGRESSION_PAIRS_DATA_PATH, \
    LABELED_PROGRESSION_PAIRS_TRAIN_PATH, \
    LABELED_PROGRESSION_PAIRS_VALID_PATH, \
    LABELED_PROGRESSION_PAIRS_TEST_PATH, \
    LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH, \
    LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH, \
    LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH, \
    LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH, \
    BEST_SEG_NDJSON_PATH, BEST_SEG_PROGRESSION_PAIRS_PATH, \
    PRECURRENTPOST_DATAWITHANNOTATIONS_PATH, PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH, \
    VAEz_PATH
from src import utils
from src.data_manager.quickdraw import build_category_index, \
    normalize_strokes, stroke3_to_stroke5
from src.models.base.stroke_models import NdjsonStrokeDataset
from src.models.core import nn_utils, transformer_utils





##############################################################################
#
# DATASET
#
##############################################################################

PAD_ID, OOV_ID, SOS_ID, EOS_ID = 0, 1, 2, 3 # TODO: this should be a part of dataset maybe?

def build_vocab(data):
    """
    Returns mappings from index to token and vice versa.

    Args:
        data: list of dicts, each dict is one example.
    """
    tokens = set()
    for sample in data:
        text = utils.normalize_sentence(sample['annotation'])
        for token in text:
            tokens.add(token)

    idx2token = {}
    tokens = ['PAD', 'OOV', 'SOS', 'EOS'] + list(tokens)
    for i, token in enumerate(tokens):
        idx2token[i] = token
    token2idx = {v:k for k, v in idx2token.items()}

    return idx2token, token2idx


def save_progression_pair_dataset_splits_and_vocab():
    """
    Each split is a list of dicts, each dict is one example
    """
    tr_amt, val_amt, te_amt = 0.9, 0.05, 0.05

    # load data (saved by quickdraw.py)
    category_to_data = {}
    for fn in os.listdir(LABELED_PROGRESSION_PAIRS_DATA_PATH):
        category = os.path.splitext(fn)[0]  # cat.pkl
        fp = os.path.join(LABELED_PROGRESSION_PAIRS_DATA_PATH, fn)
        data = utils.load_file(fp)
        category_to_data[category] = data

    # split
    train, valid, test = [], [], []
    for category, data in category_to_data.items():
        l = len(data)
        tr_idx = int(tr_amt * l)
        val_idx = int((tr_amt + val_amt) * l)
        tr_data = data[:tr_idx]
        val_data = data[tr_idx:val_idx]
        te_data = data[val_idx:]
        train += tr_data
        valid += val_data
        test += te_data

    # save splits
    for data, fp in [(train, LABELED_PROGRESSION_PAIRS_TRAIN_PATH),
                     (valid, LABELED_PROGRESSION_PAIRS_VALID_PATH),
                     (test, LABELED_PROGRESSION_PAIRS_TEST_PATH)]:
        utils.save_file(data, fp)

    # build and save vocab
    idx2token, token2idx = build_vocab(train + valid + test)
    for data, fp in [(idx2token, LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH),
                     (token2idx, LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)]:
        utils.save_file(data, fp)

    # build and save category to index map (in case our model conditions on category)
    idx2cat, cat2idx = build_category_index(train + valid + test)
    for data, fp, in [(idx2cat, LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH),
                      (cat2idx, LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)]:
        utils.save_file(data, fp)


def map_sentence_to_index(sentence, token2idx):
    return [int(token2idx[tok]) for tok in utils.normalize_sentence(sentence)]


def data_augmentation_on_instruction(text):
    """
    Args:
        text (str): instruction

    Returns:
        str
    """
    # swap "draw" / "add"
    text_tokens = nltk.word_tokenize(text)
    for i, token in enumerate(text_tokens):
        if token in ['draw', 'add']:
            text_tokens[i] = random.choice(['draw', 'add'])
    text = ' '.join(text_tokens)

    # shuffle order of sentences
    text = nltk.sent_tokenize(text)
    random.shuffle(text)
    text = ' '.join(text)

    return text

def save_drawingsasimages_annotated_dataset_splits():
    """
    Each split is a list of dicts, each dict is one example
    """
    tr_amt, val_amt, te_amt = 0.9, 0.05, 0.05

    # load data (saved by quickdraw.py)
    category_to_data = {}
    for fn in os.listdir(PRECURRENTPOST_DATAWITHANNOTATIONS_PATH):
        category = os.path.splitext(fn)[0]  # cat.pkl
        fp = os.path.join(PRECURRENTPOST_DATAWITHANNOTATIONS_PATH, fn)
        data = utils.load_file(fp)
        category_to_data[category] = data

    # split
    train, valid, test = [], [], []
    for category, data in category_to_data.items():
        l = len(data)
        tr_idx = int(tr_amt * l)
        val_idx = int((tr_amt + val_amt) * l)
        tr_data = data[:tr_idx]
        val_data = data[tr_idx:val_idx]
        te_data = data[val_idx:]
        train += tr_data
        valid += val_data
        test += te_data

    # save splits
    os.makedirs(PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH, exist_ok=True)
    for fn, data in {'train.pkl': train, 'valid.pkl': valid, 'test.pkl': test}.items():
        fp = os.path.join(PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH, fn)
        utils.save_file(data, fp)


class DrawingsAsImagesAnnotatedDataset(Dataset):
    """
    Annotated with instructions, drawing is represented as images
    (generated from src/data_manger/quickdraw.py's save_drawings_split_into_precurrentpost()).
    """
    def __init__(self,
                 dataset_split,
                 images='annotated',
                 data_aug_on_text=True,
                 data_aug_on_imgs=False,
                 rank_imgs_text=False, n_rank_imgs=8):
        """
        Args:
            dataset_split (str): 'train', 'valid', 'test'
            images (str): comma separated list. Possible values
                annotated,pre,post,start_to_annotated,full
            data_aug_on_text (bool): simple data augmentation of instructions
            predict_img_text (bool): predict which image (which slice of drawing) matches with annotation
        """
        super().__init__()
        self.dataset_split = dataset_split
        self.images = images.split(',')
        self.data_aug_on_text = data_aug_on_text
        self.data_aug_on_imgs = data_aug_on_imgs
        self.rank_imgs_text = rank_imgs_text
        self.n_rank_imgs = n_rank_imgs

        # Get data
        fp = None
        if dataset_split == 'train':
            fp = os.path.join(PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH, 'train.pkl')
        elif dataset_split == 'valid':
            fp =  os.path.join(PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH, 'valid.pkl')
        elif dataset_split == 'test':
            fp =  os.path.join(PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH, 'test.pkl')
        if not os.path.exists(fp):  # create splits and vocab first time
            save_drawingsasimages_annotated_dataset_splits()
        self.data = utils.load_file(fp)

        # Load vocab and category mappings
        self.idx2token = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.vocab_size = len(self.idx2token)

        self.idx2cat = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH)
        self.cat2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)

        # For data augmentation
        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.RandomAffine(0, translate=(0, 0.1)),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR)
        ])

    def __len__(self):
        return len(self.data)

    def _load_img_as_np(self, img_fp):
        return np.array(Image.open(img_fp))

    def _construct_image(self, sample):
        """
        An "image" a [N_channels, H, W] tensor where each channel is an
        image of the drawing (i.e. the annotated segment, the full drawing, etc.)

        Returns:
            [N_channel, H, W] np array
        """
        imgs = []
        if 'annotated' in self.images:
            imgs.append(self._load_img_as_np(sample['annotated_seg_fp']))
        if 'pre' in self.images:
            imgs.append(self._load_img_as_np(sample['pre_seg_fp']))
        if 'start_to_annotated' in self.images:
            imgs.append(self._load_img_as_np(sample['start_to_annotated_fp']))
        if 'post' in self.images:
            imgs.append(self._load_img_as_np(sample['post_seg_fp']))
        if 'full' in self.images:
            imgs.append(self._load_img_as_np(sample['full_fp']))
        drawing = np.stack(imgs)  # ["channels", H, W]

        if (self.dataset_split == 'train') and self.data_aug_on_imgs:
            drawing = self.img_transforms(drawing)

        return drawing

    def _construct_rank_image(self, start, end, n_segs, sample):
        """
        Construct "image" for a image that will be used in ranking loss.
        "Image" channels are various segments of the drawing.

        Returns:
            [N_channel, H, W] np array
        """
        base_dir = os.path.dirname(sample['annotated_seg_fp'])

        imgs = []
        if 'annotated' in self.images:
            fn = f'{start}-{end}.jpg'
            imgs.append(self._load_img_as_np(os.path.join(base_dir, fn)))
        if 'pre' in self.images:
            fn = f'0-{start}.jpg'
            imgs.append(self._load_img_as_np(os.path.join(base_dir, fn)))
        if 'start_to_annotated' in self.images:
            fn = f'0-{end}.jpg'
            imgs.append(self._load_img_as_np(os.path.join(base_dir, fn)))
        if 'post' in self.images:
            fn = f'{end}-{n_segs}.jpg'
            imgs.append(self._load_img_as_np(os.path.join(base_dir, fn)))
        if 'full' in self.images:
            imgs.append(self._load_img_as_np(sample['full_fp']))
        drawing = np.stack(imgs)  # ["channels", H, W]
        return drawing

    def _get_start_end_from_imgfp(self, imgfp):
        """
        Helper to to get start and end of segment for drawing segment in imgfp

        data/quickdraw/precurrentpost/data/zebra/6223547657617408/9-12.jpg' -> 9, 12
        """
        fn = os.path.basename(imgfp)
        start, end = fn.strip('.jpg').split('-')
        start, end = int(start), int(end)
        return start, end

    def _get_imgfp_from_start_end(self, start, end, sample):
        """Helper to return fp for image of segment from start to end"""
        base_dir = os.path.dirname(sample['annotated_seg_fp'])
        fn = f'{start}-{end}.jpg'
        fp = os.path.join(base_dir, fn)
        return fp

    def get_rankimgs_and_prefs(self, seg_start, seg_end, n_segs, sample):
        """
        Args:
            seg_start (int): start idx of annotated segment
            seg_end (int): end idx of annotated segment
            n_segs (int):
            sample: __getitem__ sample

        Note:
            Pref is top-1 probability of that image (probability that it should be ranked first)

        Returns:
            rank_imgs: [n_rank_imgs, C, H, W]
            prefs: [n_rank_imgs] np array
                Softmax over scores
        """

        def add_img(imgfps_pref, added_ranges, left, right, n_segs, sample, pref, allow_dups=False):
            """
            Add image (segment of drawing) to rank images

            Args:
                imgfps_pref (list of tuples): tuple is (fp to image, preference score)
                added_ranges (set): set of tuples storing which (left, right) segments have been added
                left (int): start idx of segment
                right (int): end idx of segment
                n_segs (int)
                sample (__getitem__ sample)
                pref (int): preference
                allow_dups (bool): allow duplicate segments in imgfps_pref

            Returns:
                potentially modified versions of imgfps_pref, added_ranges
            """
            # Early shortcut. Range doesn't make sense
            if (left >= right) or (left < 0) or (right > n_segs):
                return imgfps_pref, added_ranges

            # Add if segment hasn't been added before
            if ((left, right) not in added_ranges) or allow_dups:
                fp = self._get_imgfp_from_start_end(left, right, sample)
                imgfps_pref.append((fp, pref))
                added_ranges.add((left, right))

            return imgfps_pref, added_ranges

        PERFECT_PREF = 5
        DISTANT_PREF = 1

        # Add the actual annotated segment
        imgfps_prefs = [(self._get_imgfp_from_start_end(seg_start, seg_end, sample), PERFECT_PREF)]
        added_ranges = set([(seg_start, seg_end)])

        # Add some close images
        max_delta = 2
        for delta in range(1, max_delta+1):
            left, right = seg_start - delta, seg_end
            imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, left, right, n_segs, sample, PERFECT_PREF - delta)
            left, right = seg_start + delta, seg_end
            imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, left, right, n_segs, sample, PERFECT_PREF - delta)
            left, right = seg_start, seg_end - delta
            imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, left, right, n_segs, sample, PERFECT_PREF - delta)
            left, right = seg_start, seg_end + delta
            imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, left, right, n_segs, sample, PERFECT_PREF - delta)

        # Add some "distant" images (random segments) that have no overlap with the annotated segment
        # There may be duplicates in order to ensure that there is a total of self.n_rank_imgs
        # want a mix of close and distant
        n_added = 0
        left_seg_exists, right_seg_exists = False, False
        while n_added < self.n_rank_imgs:
            # Segments on the left of the annotated segment
            if (seg_start > 0):
                left_seg_exists = True
                left = random.choice(range(0, seg_start))
                right = random.choice(range(left + 1, seg_start + 1))
                imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, left, right, n_segs, sample, DISTANT_PREF, allow_dups=True)
                n_added += 1

            # Segments on the right of the annotated segment
            if (seg_end < n_segs):
                right_seg_exists = True
                left = random.choice(range(seg_end, n_segs))
                right = random.choice(range(left + 1, n_segs + 1))
                imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, left, right, n_segs, sample, DISTANT_PREF, allow_dups=True)
                n_added += 1

            if (not left_seg_exists) and (not right_seg_exists):
                break

        # If there aren't any distant segments (from above), we may have to sample within the annotated segment
        # / add the annotated segment itself
        # For example, full drawing is just 0-1.jpg, and segment is 0-1
        if (not left_seg_exists) and (not right_seg_exists):
            while len(imgfps_prefs) < self.n_rank_imgs:
                # Note: not sampling within right now.
                imgfps_pref, added_ranges = add_img(imgfps_prefs, added_ranges, seg_start, seg_end, n_segs, sample, PERFECT_PREF, allow_dups=True)

        # Get n_rank_imgs (actually annotated image that we want the model to rank first, and other images)
        annotated_imgfp_pref = imgfps_prefs[0]
        other_imgfps_prefs = imgfps_prefs[1:]
        random.shuffle(other_imgfps_prefs)
        imgfps_prefs = [annotated_imgfp_pref] + other_imgfps_prefs[:self.n_rank_imgs - 1]    # select (n_rank_imgs - 1) random other imgs


        # Shuffle them so it's not always the first image that is weighted highest
        random.shuffle(imgfps_prefs)
        imgfps, prefs = zip(*imgfps_prefs)

        # Load the actual images and combine them into one tensor
        rank_imgs = []
        for imgfp in imgfps:
            start, end = self._get_start_end_from_imgfp(imgfp)
            rank_img = self._construct_rank_image(start, end, n_segs, sample)
            rank_imgs.append(rank_img)
        rank_imgs = np.stack(rank_imgs)  # [n_rank_imgs, C, H, W]

        # Softmax
        prefs = np.array(prefs)
        prefs = np.exp(prefs) / np.sum(np.exp(prefs), axis=0)

        return rank_imgs, prefs


    def __getitem__(self, idx):
        sample = self.data[idx]
        drawing = self._construct_image(sample)  # [C, H, W] np array

        # Images to be ranked against generated instruction embedding
        rank_imgs = np.array([1])  # dummy (not None because it's stacked in the collate_fn)
        rank_imgs_pref = np.array([1])  # dummy (not None because it's stacked in the collate_fn)
        if self.rank_imgs_text:
            _, n_segs = self._get_start_end_from_imgfp(sample['post_seg_fp'])
            seg_start, seg_end = self._get_start_end_from_imgfp(sample['annotated_seg_fp'])
            rank_imgs, rank_imgs_pref = self.get_rankimgs_and_prefs(seg_start, seg_end, n_segs, sample)

        # Get text
        text = sample['annotation']
        text = text.lower()
        if (text != '?') and (not text.endswith('.')):  # every sentence ends with period
            text = text + '.'

        if self.data_aug_on_text:
            text = data_augmentation_on_instruction(text)

        text_indices = map_sentence_to_index(text, self.token2idx)
        text_indices = [SOS_ID] + text_indices + [EOS_ID]

        # Additional metadata
        cat = sample['category']
        cat_idx = self.cat2idx[cat]
        url = sample['url']

        return (drawing, rank_imgs, rank_imgs_pref, text, text_indices, cat, cat_idx, url)

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch

        Note: I wrote my own collate_fn in order to handle variable lengths. The StrokeDataset
        uses the default collate_fn because each drawing is padded to some maximum length (this is
        how Magenta did it as well).


        Args:
            batch: list of samples, one sample is returned from __getitem__(idx)
        """
        imgs, rank_imgs, rank_imgs_pref, texts, texts_indices, cats, cats_idx, urls = zip(*batch)
        bsz = len(batch)
        # sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        # Create array of text indices, zeros for padding
        text_lens = [len(t) for t in texts_indices]
        max_text_len = max(text_lens)
        batch_text_indices = np.zeros((bsz, max_text_len))
        for i, text_indices in enumerate(texts_indices):
            l = len(text_indices)
            batch_text_indices[i,:l] = text_indices

        # Convert to Tensors
        batch_imgs = torch.FloatTensor(np.stack(imgs))
        batch_text_indices = torch.LongTensor(batch_text_indices)
        cats_idx = torch.LongTensor(cats_idx)

        batch_rank_imgs = torch.FloatTensor(np.stack(rank_imgs))  # [bsz, n_rank_imgs, C, H, W]
        batch_rank_imgs_pref = torch.FloatTensor(np.stack(rank_imgs_pref))  # [bsz, n_rank_imgs]

        # Returning rank data as a tuple to match the API of ProgressionPairDataset...
        return batch_imgs, (batch_rank_imgs, batch_rank_imgs_pref), \
            texts, text_lens, batch_text_indices, cats, cats_idx, urls


class ProgressionPairDataset(Dataset):
    def __init__(self,
                 dataset_split,
                 use_prestrokes=False,
                 use_full_drawings=False,
                 data_aug_on_text=True,
                 max_length=200,
                 ):
        """
        TODO: should add a maximum length

        Annotated dataset of segments of drawings.

        Args:
            dataset_split (str): 'train', 'valid', 'test'
            use_prestrokes (bool): concatenate strokes that occurred before the annotated segment
            use_full_drawings (bool): return the entire drawing, not just the annotated segment
        """
        super().__init__()
        self.dataset_split = dataset_split
        self.use_prestrokes = use_prestrokes
        self.use_full_drawings = use_full_drawings
        self.data_aug_on_text = data_aug_on_text

        # Get data
        fp = None
        if dataset_split == 'train':
            fp = LABELED_PROGRESSION_PAIRS_TRAIN_PATH
        elif dataset_split == 'valid':
            fp = LABELED_PROGRESSION_PAIRS_VALID_PATH
        elif dataset_split == 'test':
            fp = LABELED_PROGRESSION_PAIRS_TEST_PATH
        if not os.path.exists(fp):  # create splits and vocab first time
            save_progression_pair_dataset_splits_and_vocab()
        data = utils.load_file(fp)

        # filter
        if max_length:
            data = [d for d in data if (len(d['stroke3']) <= max_length)]

        # Load vocab and category mappings
        self.idx2token = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.vocab_size = len(self.idx2token)

        self.idx2cat = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH)
        self.cat2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)

        # TODO: should I be using stroke3_SEGMENT for the factor or stroke3? or
        # pass in the factor computed on the entire dataset?
        # TODO: Probably should just pass ins cale factor on entire sketch rnn data (which is already precomputed
        # and in stroke_models.py
        self.data = normalize_strokes(data,
                                      scale_factor_key='stroke3_segment',
                                      stroke_keys=['stroke3', 'stroke3_segment'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get subsequence of drawing that was annotated. Or full  drawing
        if self.use_full_drawings:
            stroke3 = sample['stroke3']
        else:
            stroke3 = sample['stroke3_segment']
        stroke_len = len(stroke3)
        stroke5 = stroke3_to_stroke5(stroke3)
        stroke_len += 1  # during stroke 5 conversion, there is a end of drawing point

        if self.use_prestrokes:
            # Get subsequence that precedes the annotated
            stroke3_pre = sample['stroke3'][:sample['stroke3_start'],:]
            # Insert element so that all presegments have length at least 1
            stroke3_pre = np.vstack([np.array([0, 0, 1]), stroke3_pre])  # 1 for penup

            # Separator point is [0,0,1,1,0]. Should be identifiable as pt[2] is pen down, pt[3] is pen up and
            # doesn't occur in the data otherwise
            # TODO: is this a good separator token?
            sep_pt = np.array([0,0,1,1,0])
            stroke5_pre = stroke3_to_stroke5(stroke3_pre)
            stroke5 = np.vstack([stroke5_pre, sep_pt, stroke5])

        # Map
        text = sample['annotation']
        text = text.lower()
        # every sentence ends with period
        if (text != '?') and (not text.endswith('.')):
            text = text + '.'

        if self.data_aug_on_text:
            text = data_augmentation_on_instruction(text)

        text_indices = map_sentence_to_index(text, self.token2idx)
        text_indices = [SOS_ID] + text_indices + [EOS_ID]

        # Additional metadata
        cat = sample['category']
        cat_idx = self.cat2idx[cat]
        url = sample['url']

        return (stroke5, stroke_len, text, text_indices, cat, cat_idx, url)

    @staticmethod
    def collate_fn_strokes_categories_only(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch

        When this is used, dataloader will only return the strokes, stroke lengths,
        categories (strings), and category indices. This is sort of a hack to make it compatabile
        with the StrokeDatasets, which return those 4 items.

        Args:
            batch: list of samples, one sample is returned from __getitem__(idx)
        """
        strokes, stroke_lens, texts, texts_indices, cats, cats_idx, urls = zip(*batch)
        bsz = len(batch)
        sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        # Create array of strokes, zeros for padding
        max_stroke_len = max(stroke_lens)
        batch_strokes = np.zeros((bsz, max_stroke_len, sample_dim))
        for i, stroke in enumerate(strokes):
            l = stroke_lens[i]
            batch_strokes[i,:l,:] = stroke

        # Convert to Tensors
        batch_strokes = torch.FloatTensor(batch_strokes)
        stroke_lens = torch.LongTensor(stroke_lens)
        cats_idx = torch.LongTensor(cats_idx)

        return batch_strokes, stroke_lens, cats, cats_idx

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch

        Note: I wrote my own collate_fn in order to handle variable lengths. The StrokeDataset
        uses the default collate_fn because each drawing is padded to some maximum length (this is
        how Magenta did it as well).


        Args:
            batch: list of samples, one sample is returned from __getitem__(idx)
        """
        strokes, stroke_lens, texts, texts_indices, cats, cats_idx, urls = zip(*batch)
        bsz = len(batch)
        sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        # Create array of strokes, zeros for padding
        max_stroke_len = max(stroke_lens)
        batch_strokes = np.zeros((bsz, max_stroke_len, sample_dim))
        for i, stroke in enumerate(strokes):
            l = stroke_lens[i]
            batch_strokes[i,:l,:] = stroke

        # Create array of text indices, zeros for padding
        text_lens = [len(t) for t in texts_indices]
        max_text_len = max(text_lens)
        batch_text_indices = np.zeros((bsz, max_text_len))
        for i, text_indices in enumerate(texts_indices):
            l = len(text_indices)
            batch_text_indices[i,:l] = text_indices

        # Convert to Tensors
        batch_strokes = torch.FloatTensor(batch_strokes)
        batch_text_indices = torch.LongTensor(batch_text_indices)
        cats_idx = torch.LongTensor(cats_idx)

        return batch_strokes, stroke_lens, \
            texts, text_lens, batch_text_indices, cats, cats_idx, urls

#
# Dataset for two-stage models
#

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

class SketchWithPlansDataset(Dataset):
    def __init__(self,
                 dataset='progressionpair',
                 max_len=200,
                 categories='all',
                 max_per_category=250,  # used with dataset='ndjson'
                 dataset_split='train',
                 instruction_set='toplevel',
                 prob_threshold=0,
                 ):
        """
        Args:
            dataset (str): 'progressionpair'
            max_len (int): maximum length of drawing
            max_per_category (int): used when dataset=='ndjson', as there are tens of thousands of examples
                                    per category
            dataset_split (str): 'train', 'valid', 'test'
            instruction_set (str):
                'toplevel': only use instruction generated for entire drawing
                'toplevel_leaves': use toplevel and all leaf instructions
                'leaves': use only leaves
            prob_threshold (float): used to prune instruction trees
        """
        # TODO: pass in categories

        self.dataset = dataset
        self.max_len = max_len
        self.categories = categories
        self.max_per_category = max_per_category
        self.dataset_split = dataset_split
        self.instruction_set = instruction_set
        self.prob_threshold = prob_threshold

        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.idx2token = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.vocab_size = len(self.token2idx)

        if dataset == 'progressionpair':
            self.ds = ProgressionPairDataset(dataset_split, use_prestrokes=False, use_full_drawings=True, max_length=max_len)
            # TODO: this is hardcoded in here, should be moved to some config
            self.plans_dir = BEST_SEG_PROGRESSION_PAIRS_PATH / dataset_split
            self.id_to_plan = self.load_progression_pair_plans(self.plans_dir)
        elif dataset == 'ndjson':
            self.ds = NdjsonStrokeDataset(categories, dataset_split,
                                          max_per_category=max_per_category, max_len=max_len, must_have_instruction_tree=True)
            # compared to progressionpair, we don't pre-load the plans because that would be too much memory
            # also, the directory is in a different format (no dataset_split)
            # TODO: this is hardcoded in here, should be moved to some config
            self.plans_dir = BEST_SEG_NDJSON_PATH

    def get_underlying_ds_item(self, idx):
        """
        Returns the same data regardless of the dataset.
        """
        if self.dataset == 'progressionpair':
            stroke5, stroke_len, _, _, cat, cat_idx, url = self.ds.__getitem__(idx)  # the _ are the ground-truth annotations for a segment of the drawing
            id = self.ds.data[idx]['id']
            plan = self.id_to_plan[id]
            plan = prune_seg_tree(plan, prob_threshold=self.prob_threshold)
            return stroke5, stroke_len, cat, cat_idx, url, plan
        elif self.dataset == 'ndjson':
            stroke5, stroke_len, cat, cat_idx = self.ds.__getitem__(idx, pad_to_max_len_in_data=False)  # _ is stroke_len
            id = self.ds.data[idx]['id']
            plan_fp = self.plans_dir / cat / f'{id}.json'
            plan = utils.load_file(plan_fp)
            plan = prune_seg_tree(plan, prob_threshold=self.prob_threshold)
            return stroke5, stroke_len, cat, cat_idx, '', plan

    def load_progression_pair_plans(self, plans_dir):
        """
        Return dict from example id (id originally found in ndjson files) to json of instruction tree plans
        produced by a trained InstructionGen model in segmentation.py.

        Args:
            plans_dir (str)
        """
        id_to_plan = {}
        for fn in os.listdir(plans_dir):
            if fn.endswith('json'):
                fp = os.path.join(plans_dir, fn)
                category, id = fn.replace('.json', '').split('_')
                plans = utils.load_file(fp)
                id_to_plan[id] = plans
        return id_to_plan

    def __len__(self):
        return len(self.ds.data)

class InstructionVAEzDataset(SketchWithPlansDataset):
    def __init__(self,
                 dataset='ndjson',
                 max_len=200,
                 categories='pig',
                 max_per_category=250,
                 dataset_split='train',
                 instruction_set='toplevel',
                 prob_threshold=0.0,
                 ):
        super().__init__(dataset=dataset, max_len=max_len,
                         categories=categories, max_per_category=max_per_category,
                         dataset_split=dataset_split, instruction_set=instruction_set,
                         prob_threshold=prob_threshold)

    def __getitem__(self, idx):
        stroke5, stroke_len, cat, cat_idx, url, plan = self.get_underlying_ds_item(idx)

        # Get instruction comprised of leaf instructions
        text = ''
        text_indices = []
        for subplan in plan:
            if (subplan['right'] - subplan['left']) == 1:  # leaf
                # TODO: ideally we should have a different separator token...
                text += ' SOS ' + subplan['text']
                text_indices += [SOS_ID] + map_sentence_to_index(subplan['text'], self.token2idx)
        text += ' EOS'
        text_indices += [EOS_ID]
        text = text.lstrip()  # get rid of leading space

        # Get the z for this drawing
        drawing_id = self.ds.data[idx]['id']
        fp = os.path.join(VAEz_PATH.format(cat, cat), f'{drawing_id}.pkl')
        vae_z = utils.load_file(fp)  # numpy
        vae_z = torch.FloatTensor(vae_z)

        return text, text_indices, cat, cat_idx, vae_z

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch

        Note: I wrote my own collate_fn in order to handle variable lengths. The StrokeDataset
        uses the default collate_fn because each drawing is padded to some maximum length (this is
        how Magenta did it as well).


        Args:
            batch: list of samples, one sample is returned from __getitem__(idx)

        Returns:
            texts: list of length bsz of strs
            text_lens: list of ints
            batch_text_indices: [max_len, bsz] LongTensor
            cats: list of strs
            cats_idx: [bsz] LongTensor
            vae_zs: [bsz, z_dim] FloatTensor
        """
        texts, texts_indices, cats, cats_idx, vae_zs = zip(*batch)
        bsz = len(batch)

        # Create array of text indices, zeros for padding
        text_lens = [len(t) for t in texts_indices]
        max_text_len = max(text_lens)
        batch_text_indices = np.zeros((bsz, max_text_len))
        for i, text_indices in enumerate(texts_indices):
            l = len(text_indices)
            batch_text_indices[i,:l] = text_indices

        # Convert to Tensors
        batch_text_indices = torch.LongTensor(batch_text_indices)
        cats_idx = torch.LongTensor(cats_idx)
        vae_zs = torch.stack(vae_zs) # [bsz, z_dim]

        batch_text_indices = batch_text_indices.transpose(0,1)  # [max_len, bsz]  # decoder expects this format
        batch_text_indices = nn_utils.move_to_cuda(batch_text_indices)
        cats_idx = nn_utils.move_to_cuda(cats_idx)
        vae_zs = nn_utils.move_to_cuda(vae_zs)

        return texts, text_lens, batch_text_indices, cats, cats_idx, vae_zs


class SketchWithPlansConditionEntireDrawingDataset(SketchWithPlansDataset):
    """
    ConditionEntireDrawing refers to how this dataset will be used. The instructions provided
    will be used embedded once and used for the entire drawing. This is in contrast to
    SketchWithPlansConditionSegmentsDataset, where segments of the drawing are conditioned
    on different stacks of instructions.
    """
    def __init__(self,
                 dataset='ndjson',
                 max_len=200,
                 max_per_category=250,
                 dataset_split='train',
                 instruction_set='toplevel',
                 prob_threshold=0.0
                 ):
        super().__init__(dataset=dataset, max_len=max_len, max_per_category=max_per_category,
                         dataset_split=dataset_split, instruction_set=instruction_set,
                         prob_threshold=prob_threshold)

    def __getitem__(self, idx):
        stroke5, stroke_len, cat, cat_idx, url, plan = self.get_underlying_ds_item(idx)

        if self.instruction_set == 'toplevel':
            text = plan[0]['text']  # 0 = toplevel instruction
            text_indices = map_sentence_to_index(text, self.token2idx)

        elif self.instruction_set == 'toplevel_leaves':
            text = plan[0]['text']
            text_indices = map_sentence_to_index(text, self.token2idx)
            for subplan in plan[1:]:
                if (subplan['right'] - subplan['left']) == 1:  # leaf
                    # TODO: ideally we should have a different separator token...
                    text += ' SOS ' + subplan['text']
                    text_indices += [SOS_ID] + map_sentence_to_index(subplan['text'], self.token2idx)

        elif self.instruction_set == 'leaves':
            text = ''
            text_indices = []
            for subplan in plan[1:]:
                if (subplan['right'] - subplan['left']) == 1:  # leaf
                    # TODO: ideally we should have a different separator token...
                    text += ' SOS ' + subplan['text']
                    text_indices += [SOS_ID] + map_sentence_to_index(subplan['text'], self.token2idx)
            text = text.lstrip()  # get rid of leading space

        return (stroke5, stroke_len, text, text_indices, cat, cat_idx, url)


class SketchWithPlansConditionSegmentsDataset(SketchWithPlansDataset):
    """
    ConditionSegments refers to how each segment in the sketch model will be conditioned
    on a different stack of instructions, the stack being from leaf to root in the
    instruction tree.
    """
    def __init__(self,
                 dataset='ndjson',
                 max_len=200,
                 max_per_category=250,
                 dataset_split='train',
                 instruction_set='stack',  # 'stack' or 'stack_leaves'
                 prob_threshold=0.0,
                 ):
        super().__init__(dataset=dataset, max_len=max_len, max_per_category=max_per_category,
                         dataset_split=dataset_split, instruction_set=instruction_set,
                         prob_threshold=prob_threshold)

    def __getitem__(self, idx):
        """
        Note: transformation into text_indices, lengths, etc. is done in collate_fn
        """
        stroke5, stroke_len, cat, cat_idx, url, plan = self.get_underlying_ds_item(idx)
        stacks = self.get_stacks(plan)

        return (stroke5, stroke_len, stacks, cat, cat_idx, url)

    @staticmethod
    def collate_fn(batch, token2idx=None):
        """
        Note: this is similar to ProgressionPair's collate_fn

        Args:
            batch: list of items from __getitem__(idx)
            token2idx: passed in using functools.partial

        Returns:
            batch_strokes ([bsz, max_seq_len, 5])
            ...
            batch_text_indices, ([bsz, max_seq_len, max_instruction_len])
            batch_text_lens ([bsz, max_seq_len]):
                length of each instruction stack
            batch_texts (list of lists): just used for debugging
            ...
        """
        strokes, stroke_lens, stacks, cats, cats_idx, urls = zip(*batch)  # each is a list
        bsz = len(batch)
        sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        #
        # Create array of strokes, zeros for padding
        #
        max_stroke_len = max(stroke_lens)
        batch_strokes = np.zeros((bsz, max_stroke_len, sample_dim))
        for i, stroke in enumerate(strokes):
            l = stroke_lens[i]
            batch_strokes[i,:l,:] = stroke

        #
        # Create array for instructions (vocab indices)
        #

        # First, get a) the maximum instruction length, b) the instruction vocab indices,
        # c) the instruction lengths
        max_text_len = -1
        batch_text_indices_list = []  # list of lists of lists
        batch_text_lens_list = []  # list of lists
        batch_texts = []
        for i in range(bsz):  # for each drawing
            drawing_text_indices = []
            drawing_text_lens = []
            drawing_texts = []
            for key, stack in stacks[i].items():
                # key is left and right indices denoting one segment, i.e. (2,3)
                # stack is a list of strings starting from the top-level instruction
                text = ' '.join(stack)  # TODO: use a separator token?
                text_indices = map_sentence_to_index(text, token2idx)
                if len(text_indices) == 0:  # TODO: this is kind of hacky... few instructions may be empty string?
                    text_indices = [EOS_ID]

                text_len = len(text_indices)
                max_text_len = max(max_text_len, text_len)
                drawing_text_indices.append(text_indices)
                drawing_text_lens.append(text_len)
                drawing_texts.append(text)
            batch_text_indices_list.append(drawing_text_indices)
            batch_text_lens_list.append(drawing_text_lens)
            batch_texts.append(drawing_texts)

        # Next, convert text_indices to an array
        batch_text_indices = np.zeros((bsz, max_stroke_len, max_text_len))
        batch_text_lens = np.zeros((bsz, max_stroke_len))
        batch_text_lens.fill(1)
        # NOTE: lengths are filled with 1. This simply avoids having 0's, even
        # for the padding elements (i.e. beyond the length of drawing i).
        # This is a bit of a hack because otherwise nans are produced with the
        # transformer encoder module. Functionaly, it shouldn't matter because
        # extra values produced for the additional padding timesteps are
        # ignored by the decoder -- in a nn.LSTM case, using the pack_padded_sequence
        # to take in the lengths; in the GMMDecoder, by masking out the targets.
        for i in range(bsz):
            # Break drawing into segments so that we can map instruction stacks to
            # corresponding part in drawing.
            stroke = strokes[i]
            pen_up = np.where(stroke[:,3] == 1)[0].tolist()  # use this to
            pen_up = ([0] + pen_up) if (pen_up[0] != 0) else pen_up  # first element could already be 0

            # keep track of which segment we're currently in. There are as many
            # stacks as there are segments.
            cur_seg_idx = 0
            for j in range(len(stroke) -1 ):  # -1 because last point is just the final [0,0,0,0,1]
                cur_seg_end = pen_up[cur_seg_idx + 1]
                if (j > cur_seg_end):
                    cur_seg_idx += 1

                stack_text_indices = batch_text_indices_list[i][cur_seg_idx]
                batch_text_indices[i,j,:len(stack_text_indices)] = stack_text_indices

                # update length
                batch_text_lens[i,j] = batch_text_lens_list[i][cur_seg_idx]

        # Convert to appropriate data format
        batch_strokes = torch.FloatTensor(batch_strokes)
        batch_text_indices = torch.LongTensor(batch_text_indices)
        batch_text_lens = torch.LongTensor(batch_text_lens)
        cats_idx = torch.LongTensor(cats_idx)

        return batch_strokes, stroke_lens, \
            batch_texts, batch_text_lens, batch_text_indices, cats, cats_idx, urls

    def get_stacks(self, plan):
        """
        Args:
            plan (list of dicts): instruction tree

        Returns:
            dict:
                key: tuples (left, right)
                value: list of texts (stack of instructions)
        """
        # initialize stacks. each stack is a list, one stack per segment
        stacks = {}
        for left in range(plan[0]['right']):
            stacks[(left, left+1)] = []

        for i, subplan in enumerate(plan):
            # add this subplan to all relevant stacks
            for left in range(subplan['left'], subplan['right']):
                if self.instruction_set == 'stack_leaves':  # only add if
                    if (subplan['right'] - subplan['left']) == 1:  # leaf
                       stacks[(left, left+1)].append(subplan['text'])
                else:
                    stacks[(left, left+1)].append(subplan['text'])

        return stacks

##############################################################################
#
# MODEL
#
##############################################################################

class InstructionEncoderTransformer(nn.Module):
    def __init__(self,
                 hidden_dim, num_layers=1, dropout=0,
                 use_categories=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_categories = use_categories

        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, 2, dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu'
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)

        if use_categories:
            self.dropout_mod = nn.Dropout(dropout)
            self.instruction_cat_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,
                text_indices, text_lens, text_embedding,
                category_embedding=None, categories=None):
        """
        Args:
            text_indices:  [max_len, bsz]
            text_lens: [bsz] (maximum value should be max_len)
            text_embedding: nn.Embedding(vocab_size, dim)
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor

        Returns:
            hidden: [bsz, dim]
        """
        bsz = text_indices.size(1)

        text_embs = text_embedding(text_indices)  # [len, bsz, dim]

        if self.use_categories:
            cats_emb =  category_embedding(categories)  # [bsz, dim]
            cats_emb = self.dropout_mod(cats_emb)
            instructions = torch.cat([instructions, cats_emb.repeat(instructions.size(0), 1, 1)], dim=2)  # [len, bsz, input+hidden]
            instructions = self.instruction_cat_fc(instructions)  # [len, bsz, hidden]

        instructions_pad_mask, _, _ = transformer_utils.create_transformer_padding_masks(src_lens=text_lens)
        memory = self.enc(text_embs, src_key_padding_mask=instructions_pad_mask)  # [len, bsz, dim]

        hidden = []
        for i in range(bsz):  # TODO: what is a tensor op to do this?
            item_len = text_lens[i]
            item_emb = memory[:item_len,i,:].mean(dim=0)  # [dim]
            hidden.append(item_emb)
        hidden = torch.stack(hidden, dim=0)  # [bsz, dim]

        return hidden

class InstructionDecoderLSTM(nn.Module):
    def __init__(self,
                 input_dim, hidden_dim,
                 num_layers=1, dropout=0, batch_first=True,
                 condition_on_hc=False, use_categories=False,
                 use_layer_norm=False, rec_dropout=0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.condition_on_hc = condition_on_hc
        self.use_categories = use_categories
        self.use_layer_norm = use_layer_norm
        self.rec_dropout = rec_dropout

        self.dropout_mod = nn.Dropout(dropout)
        # if use_layer_norm:
        #     self.lstm = haste.LayerNormLSTM(input_size=input_dim, hidden_size=hidden_dim, zoneout=dropout, dropout=rec_dropout)
        # else:
        # TODO: no layernorm in decoder yet (changing implementation)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers, dropout=dropout, batch_first= batch_first)

    def forward(self, texts_emb, text_lens, hidden=None, cell=None,
                token_embedding=None, category_embedding=None, categories=None,
                mem_emb=None):
        """
        Args:
            texts_emb: [len, bsz, dim] FloatTensor
            text_lens: list of ints, length len
            hidden: [n_layers * n_directions, bsz, dim]  FloatTensor
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
            token_embedding: nn.Embedding(vocab, dim)
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor
            mem_emb: [bsz, mem_dim] FloatTensor returned by memory lookup

        Returns:
            outputs:
                if token_embedding is None: [len, bsz, dim] FloatTensor
                else: [len, bsz, vocab] FloatTensor
            hidden: [n_layers * n_directions, bsz, dim]
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
        """


        # Condition on last layer's hidden and cell on every time step by combining last hidden and cell,
        # repeating along time dimension, and concatenating with encoded texts in feature dimension
        if self.condition_on_hc:
            last_hidden, last_cell = hidden[-1, :, :], cell[-1, :, :]  # last = [bsz, dim]
            last_hc = (last_hidden + last_cell).unsqueeze(0)  # [1, bsz, dim]
            last_hc = last_hc.repeat(texts_emb.size(0), 1, 1)  # [len, bsz, dim]
            inputs_emb = torch.cat([texts_emb, last_hc], dim=2)  # [len, bsz, dim * 2]
        else:
            inputs_emb = texts_emb

        if self.use_categories and category_embedding:
            cats_emb = category_embedding(categories)  # [bsz, dim]
            cats_emb = self.dropout_mod(cats_emb)
            # repeat embedding along time dimension
            cats_emb = cats_emb.repeat(inputs_emb.size(0), 1, 1)  # [len, bsz, dim]
            inputs_emb = torch.cat([inputs_emb, cats_emb], dim=2)  # [len, bsz, dim * 2 or dim *3]

        if mem_emb is not None:
            # repeat embedding along time dimension
            mem_emb = mem_emb.repeat(inputs_emb.size(0), 1, 1)  # [len, bsz, mem_dim]
            inputs_emb = torch.cat([inputs_emb, mem_emb], dim=2)  # [len, bsz, ...]

        # decode
        if self.use_layer_norm:
            # Will have to mask out using text_lens later during loss
            outputs, (hidden, cell) = self.lstm(inputs_emb, (hidden, cell))
        else:
            packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs_emb, text_lens, enforce_sorted=False)
            outputs, (hidden, cell) = self.lstm(packed_inputs, (hidden, cell))
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            # [max_text_len + 2, bsz, dim]; h/c = [n_layers * n_directions, bsz, dim]

        if token_embedding is not None:
            outputs = torch.matmul(outputs, token_embedding.weight.t())  # [len, bsz, vocab]

        return outputs, (hidden, cell)

    def generate(self,
                 token_embedding,
                 category_embedding=None, categories=None,
                 mem_emb=None,
                 init_ids=None, hidden=None, cell=None,
                 pad_id=None, eos_id=None,
                 max_len=100,
                 decode_method=None, tau=None, k=None,
                 idx2token=None,
                 ):
        """
        Decode up to max_len symbols by feeding previous output as next input.

        Args:
            lstm: nn.LSTM
            token_embedding: nn.Embedding(vocab, dim)
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor
            mem_emb ([bsz, mem_dim]): returned by memory lookup
            init_ids:   # [init_len, bsz]
            init_embs: [init_len, bsz, emb] (e.g. embedded SOS ids)
            hidden: [layers * direc, bsz, dim]
            cell: [layers * direc, bsz, dim]
            condition_on_hc: bool (condition on hidden and cell every time step)
            EOS_ID: int (id for EOS_ID token)
            decode_method: str (how to sample words given probabilities; 'greedy', 'sample')
            tau: float (temperature for softmax)
            k: int (for sampling or beam search)
            idx2token: dict
            cats_emb: [bsz, dim]

       Returns:
            decoded_probs: [bsz, max_len, vocab]
            decoded_ids: [bsz, max_len]
            decoded_texts: list of strs
        """
        init_len, bsz = init_ids.size()
        vocab_size = len(idx2token)

        # Track which sequences have generated eos_id
        rows_with_eos = nn_utils.move_to_cuda(torch.zeros(bsz).long())
        pad_ids = nn_utils.move_to_cuda(torch.Tensor(bsz).fill_(pad_id)).long()
        pad_prob = nn_utils.move_to_cuda(torch.zeros(bsz, vocab_size))  # one hot for pad id
        pad_prob[:, pad_id] = 1

        # Generate
        decoded_probs = nn_utils.move_to_cuda(torch.zeros(bsz, max_len, vocab_size))  #
        decoded_ids = nn_utils.move_to_cuda(torch.zeros(bsz, max_len).long())  # [bsz, max_len]
        cur_input_id = init_ids

        # unsqueeze in time dimension
        cats_emb = category_embedding(categories).unsqueeze(0) if (category_embedding is not None) else None  # [1, bsz, dim]
        mem_emb = mem_emb.unsqueeze(0) if (mem_emb is not None) else None  # [1, bsz, mem_dim]

        for t in range(max_len):
            cur_input_emb = token_embedding(cur_input_id)  # [1, bsz, dim]
            if self.condition_on_hc:
                last_hc = hidden[-1, :, :] + cell[-1, :, :]  # [bsz, dim]
                last_hc = last_hc.unsqueeze(0)  # [1, bsz, dim]
                cur_input_emb = torch.cat([cur_input_emb, last_hc], dim=2)  # [1, bsz, dim * 2]
            if (cats_emb is not None):
                cur_input_emb = torch.cat([cur_input_emb, cats_emb], dim=2)  # [1, bsz, dim * 2 or dim * 3]
            if (mem_emb is not None):
                cur_input_emb = torch.cat([cur_input_emb, mem_emb], dim=2)  # [1, bsz, ...]

            dec_outputs, (hidden, cell) = self.lstm(cur_input_emb, (hidden, cell))  # [cur_len, bsz, dim]; h/c

            # Compute logits over vocab, use last output to get next token
            # TODO: can we use self.forward
            logits = torch.matmul(dec_outputs, token_embedding.weight.t())  # [cur_len, bsz, vocab]
            logits.transpose_(0, 1)  # [bsz, cur_len, vocab]
            logits = logits[:, -1, :]  # last output; [bsz, vocab]
            prob = nn_utils.logits_to_prob(logits, tau=tau)  # [bsz, vocab]
            prob, ids = nn_utils.prob_to_vocab_id(prob, decode_method, k=k)  # prob: [bsz, vocab]; ids: [bsz, k]
            ids = ids[:, 0]  # get top k; [bsz]

            # Update generated sequence so far
            # If sequence (row) has already produced an eos_id *earlier*, replace id/prob with pad
            # TODO: I don't think decoded_probs is being filled with pad_prob for some reason
            prob = torch.where((rows_with_eos == 1).unsqueeze(1), pad_prob, prob)  # unsqueeze to broadcast
            ids = torch.where(rows_with_eos == 1, pad_ids, ids)
            decoded_probs[:, t, :] = prob
            decoded_ids[:, t] = ids

            # Update for next iteration in loop
            rows_with_eos = rows_with_eos | (ids == eos_id).long()
            cur_input_id = ids.unsqueeze(0)  # [1, bsz]

            # Terminate early if all sequences have generated eos
            if rows_with_eos.sum().item() == bsz:
                break

        # TODO: sort out init wonkiness
        # Remove initial input to decoder
        # decoded_probs = decoded_probs[:, init_embs.size(1):, :]
        # decoded_ids = decoded_ids[:, init_embs.size(1):]

        # Convert to strings
        decoded_texts = []
        if idx2token is not None:
            for i in range(bsz):
                tokens = []
                for j in range(decoded_ids.size(1)):
                    id = decoded_ids[i][j].item()
                    if id == eos_id:
                        break
                    tokens.append(idx2token[id])
                text = ' '.join(tokens)
                decoded_texts.append(text)

        return decoded_probs, decoded_ids, decoded_texts