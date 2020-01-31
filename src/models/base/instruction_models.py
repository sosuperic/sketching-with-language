# instruction_models.py

"""
Instruction (annotations from MTurk) related models and dataset
"""

import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from config import LABELED_PROGRESSION_PAIRS_DATA_PATH, \
    LABELED_PROGRESSION_PAIRS_TRAIN_PATH, \
    LABELED_PROGRESSION_PAIRS_VALID_PATH, \
    LABELED_PROGRESSION_PAIRS_TEST_PATH, \
    LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH, \
    LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH, \
    LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH, \
    LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH, \
    BEST_SEG_NDJSON_PATH, BEST_SEG_PROGRESSION_PAIRS_PATH
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


class ProgressionPairDataset(Dataset):
    def __init__(self,
                 dataset_split,
                 use_prestrokes=False,
                 use_full_drawings=False,
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
class SketchWithPlansDataset(Dataset):
    def __init__(self,
                 dataset='progressionpair',
                 max_len=200,
                 max_per_category=250,  # used with dataset='ndjson'
                 dataset_split='train',
                 instruction_set='toplevel'
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
        """
        # TODO: pass in categories

        self.dataset = dataset
        self.max_len = max_len
        self.max_per_category = max_per_category
        self.dataset_split = dataset_split
        self.instruction_set = instruction_set

        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)

        if dataset == 'progressionpair':
            self.ds = ProgressionPairDataset(dataset_split, use_prestrokes=False, use_full_drawings=True, max_length=max_len)
            # TODO: this is hardcoded in here, should be moved to some config
            self.plans_dir = BEST_SEG_PROGRESSION_PAIRS_PATH / dataset_split
            self.id_to_plan = self.load_progression_pair_plans(self.plans_dir)
        elif dataset == 'ndjson':
            self.ds = NdjsonStrokeDataset('all', dataset_split,
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
            return stroke5, stroke_len, cat, cat_idx, url, plan
        elif self.dataset == 'ndjson':
            stroke5, stroke_len, cat, cat_idx = self.ds.__getitem__(idx, pad_to_max_len_in_data=False)  # _ is stroke_len
            id = self.ds.data[idx]['id']
            plan_fp = self.plans_dir / cat / f'{id}.json'
            plan = utils.load_file(plan_fp)
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
                 instruction_set='toplevel'
                 ):
        super().__init__(dataset=dataset, max_len=max_len, max_per_category=max_per_category,
                         dataset_split=dataset_split, instruction_set=instruction_set)

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
                 instruction_set='stack'
                 ):
        super().__init__(dataset=dataset, max_len=max_len, max_per_category=max_per_category,
                         dataset_split=dataset_split, instruction_set=instruction_set)

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
            pen_up = [0] + pen_up

            # keep track of which segment we're currently in. There are as many
            # stacks as there are segments.
            cur_seg_idx = 0
            for j in range(len(stroke)):
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

        # if self.use_categories:
        #     cats_emb =  category_embedding(categories)  # [bsz, dim]
        #     cats_emb = self.dropout_mod(cats_emb)
        #     instructions = torch.cat([instructions, cats_emb.repeat(instructions.size(0), 1, 1)], dim=2)  # [len, bsz, input+hidden]
        #     instructions = self.instruction_cat_fc(instructions)  # [len, bsz, hidden]

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
                 input_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True,
                 condition_on_hc=False, use_categories=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim,
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.condition_on_hc = condition_on_hc
        self.use_categories = use_categories

        self.dropout_mod = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers, dropout=dropout, batch_first= batch_first)

    def forward(self, texts_emb, text_lens, hidden=None, cell=None,
                token_embedding=None, category_embedding=None, categories=None):
        """
        Args:
            texts_emb: [len, bsz, dim] FloatTensor
            text_lens: list of ints, length len
            hidden: [n_layers * n_directions, bsz, dim]  FloatTensor
            cell: [n_layers * n_directions, bsz, dim] FloatTensor
            token_embedding: nn.Embedding(vocab, dim)
            category_embedding: nn.Embedding(n_categories, dim)
            categories: [bsz] LongTensor

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
            cats_emb = cats_emb.repeat(inputs_emb.size(0), 1, 1)  # [len, bsz, dim]
            inputs_emb = torch.cat([inputs_emb, cats_emb], dim=2)  # [len, bsz, dim * 2 or dim *3]

        # decode
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
        cats_emb = category_embedding(categories).unsqueeze(0) if (category_embedding is not None) else None  # [1, bsz, dim]
        for t in range(max_len):
            cur_input_emb = token_embedding(cur_input_id)  # [1, bsz, dim]
            if self.condition_on_hc:
                last_hc = hidden[-1, :, :] + cell[-1, :, :]  # [bsz, dim]
                last_hc = last_hc.unsqueeze(0)  # [1, bsz, dim]
                cur_input_emb = torch.cat([cur_input_emb, last_hc], dim=2)  # [1, bsz, dim * 2]
            if (cats_emb is not None):
                cur_input_emb = torch.cat([cur_input_emb, cats_emb], dim=2)  # 1, bsz, dim * 2 or dim * 3]

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