# instruction_gen.py

import argparse
import matplotlib
matplotlib.use('Agg')
from nltk.tokenize import word_tokenize  # TODO: add the download punkt to requirements.txt
import numpy as np
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.data_manager.quickdraw import LABELED_PROGRESSION_PAIRS_PATH, LABELED_PROGRESSION_PAIRS_DATA_PATH
from src.models.sketch_rnn import stroke3_to_stroke5, TrainNN
from src.models.train_nn import TrainNN, RUNS_PATH
from src.models.transformer_utils import *
import src.utils as utils

LABELED_PROGRESSION_PAIRS_TRAIN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'train.pkl')
LABELED_PROGRESSION_PAIRS_VALID_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'valid.pkl')
LABELED_PROGRESSION_PAIRS_TEST_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'test.pkl')

LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2token.pkl')
LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'token2idx.pkl')
LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'idx2cat.pkl')
LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'cat2idx.pkl')


USE_CUDA = torch.cuda.is_available()


##############################################################################
#
# HYPERPARAMETERS
#
##############################################################################
class HParams():
    def __init__(self):
        # Training
        self.batch_size = 64  # 100
        self.lr = 0.0001  # 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001  #
        self.grad_clip = 1.0
        self.max_epochs = 100

        # Model
        self.dim = 256
        self.n_enc_layers = 4
        self.n_dec_layers = 4
        self.use_pre_strokes = False
        # dropout

        # inference
        self.decode_method = 'greedy'  # 'sample', 'greedy'
        self.tau = 1.0  # sampling text
        self.k = 5      # sampling text



##############################################################################
#
# DATASET
#
##############################################################################

PAD_ID, OOV_ID, SOS_ID, EOS_ID = 0, 1, 2, 3 # TODO: this should be a part of dataset maybe?

def normalize(sentence):
    """Tokenize"""
    return word_tokenize(sentence.lower())

def build_vocab(data):
    """
    Returns mappings from index to token and vice versa.
    
    :param data: list of dicts, each dict is one example.
    """
    tokens = set()
    for sample in data:
        text = normalize(sample['annotation'])
        for token in text:
            tokens.add(token)

    idx2token = {}
    tokens = ['PAD', 'OOV', 'SOS', 'EOS'] + list(tokens)
    for i, token in enumerate(tokens):
        idx2token[i] = token
    token2idx = {v:k for k, v in idx2token.items()}

    return idx2token, token2idx

def build_category_index(data):
    """
    Returns mappings from index to category and vice versa.
    
    :param data: list of dicts, each dict is one example
    """
    categories = set()
    for sample in data:
        categories.add(sample['category'])
    idx2cat = {i: cat for i, cat in enumerate(categories)}
    cat2idx = {cat: i  for i, cat in idx2cat.items()}

    return idx2cat, cat2idx

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

def map_str_to_index(s, token2idx):
    return [int(token2idx[tok]) for tok in normalize(s)]


def normalize_data(data):
    """
    Normalize entire dataset (delta_x, delta_y) by the scaling factor.

    :param data: list of dicts
    """
    scale_factor = calculate_normalizing_scale_factor(data)
    normalized_data = []
    for sample in data:
        stroke3_seg = sample['stroke3_segment']
        stroke3 = sample['stroke3']
        stroke3_seg[:, 0:2] /= scale_factor
        stroke3[:, 0:2] /= scale_factor
        sample['stroke3_segment'] = stroke3_seg
        sample['stroke3'] = stroke3
        normalized_data.append(sample)
    return normalized_data

def calculate_normalizing_scale_factor(data):  # calculate_normalizing_scale_factor() in sketch_rnn/utils.py
    """
    Calculate the normalizing factor in Appendix of paper

    :param data: list of dicts
    """
    deltas = []
    for sample in data:
        stroke = sample['stroke3_segment']
        for j in range(stroke.shape[0]):
            deltas.append(stroke[j][0])
            deltas.append(stroke[j][1])
    deltas = np.array(deltas)
    scale_factor = np.std(deltas)
    return scale_factor


class ProgressionPairDataset(Dataset):
    def __init__(self, dataset_split, remove_question_marks=False):
        """
        
        Args:
            dataset_split: str
            remove_question_marks: bool (whether to remove samples where annotation was '?')
        """
        super().__init__()
        self.dataset_split = dataset_split
        self.remove_quesiton_marks = remove_question_marks

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

        if remove_question_marks:
            new_data = []
            for sample in data:
                if sample['annotation'] != '?':
                    new_data.append(sample)
            data = new_data

        # Load vocab and category mappings
        self.idx2token = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH)
        self.token2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH)
        self.vocab_size = len(self.idx2token)

        self.idx2cat = utils.load_file(LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH)
        self.cat2idx = utils.load_file(LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH)

        self.data = normalize_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Get subsequence of drawing that was annotated
        stroke3_seg = sample['stroke3_segment']
        stroke5_seg = stroke3_to_stroke5(stroke3_seg, len(stroke3_seg))
        # Get subsequence that precedes the annotated
        stroke3_pre_seg = sample['stroke3'][:sample['stroke3_start'],:]
        stroke5_pre_seg = stroke3_to_stroke5(stroke3_pre_seg, len(stroke3_pre_seg))

        # Map
        text = sample['annotation']
        text_indices = map_str_to_index(text, self.token2idx)
        text_indices = [SOS_ID] + text_indices + [EOS_ID]

        # Additional metadata
        cat = sample['category']
        cat_idx = self.cat2idx[cat]
        url = sample['url']

        return (stroke5_seg, stroke5_pre_seg, text, text_indices, cat, cat_idx, url)

    @staticmethod
    def collate_fn(batch):
        """
        Method to passed into a DataLoader that defines how to combine samples in a batch
        
        :param: batch: list of samples, one sample is returned from __getitem__(idx)
        """
        strokes, pre_strokes, texts, texts_indices, cats, cats_idx, urls = zip(*batch)
        bsz = len(batch)
        sample_dim = strokes[0].shape[1]  # 3 if stroke-3, 5 if stroke-5 format

        # Create array of strokes, zeros for padding
        stroke_lens = [stroke.shape[0] for stroke in strokes]
        max_stroke_len = max(stroke_lens)
        batch_strokes = np.zeros((bsz, max_stroke_len, sample_dim))
        for i, stroke in enumerate(strokes):
            l = stroke.shape[0]
            batch_strokes[i,:l,:] = stroke

        # Create array of strokes, zeros for padding
        pre_stroke_lens = [prestroke.shape[0] for prestroke in pre_strokes]
        max_pre_stroke_len = max(pre_stroke_lens)
        batch_pre_strokes = np.zeros((bsz, max_pre_stroke_len, sample_dim))
        for i, pre_stroke in enumerate(pre_strokes):
            l = pre_stroke.shape[0]
            batch_pre_strokes[i, :l, :] = pre_stroke

        # Create array of text indices, zeros for padding
        text_lens = [len(t) for t in texts_indices]
        max_text_len = max(text_lens)
        batch_text_indices = np.zeros((bsz, max_text_len))
        for i, text_indices in enumerate(texts_indices):
            l = len(text_indices)
            batch_text_indices[i,:l] = text_indices

        # Convert to Tensors
        batch_strokes = torch.FloatTensor(batch_strokes)
        batch_pre_strokes = torch.FloatTensor(batch_pre_strokes)
        batch_text_indices = torch.LongTensor(batch_text_indices)

        return batch_strokes, stroke_lens, batch_pre_strokes, pre_stroke_lens,\
               texts, text_lens, batch_text_indices, cats, cats_idx, urls



##############################################################################
#
# MODEL
#
##############################################################################


class InstructionRNN(TrainNN):
    def __init__(self, hp, save_dir):
        super().__init__(hp, save_dir)

        self.tr_loader, self.val_loader = self.get_data_loaders()

        # Model
        d_model = self.hp.dim
        self.pos_enc = PositionalEncoder(d_model, max_seq_len=250)  # [1, max_seq_len, dim]  (1 for broadcasting with bsz)
        self.strokes_input_fc = nn.Linear(5, d_model)
        self.tokens_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, dim_feedforward=d_model * 4,
            nhead=2,
            activation='relu',
            num_encoder_layers=self.hp.n_enc_layers,
            num_decoder_layers=self.hp.n_dec_layers)
        self.vocab_out_fc = nn.Linear(d_model, self.tr_loader.dataset.vocab_size)

        self.models = [self.pos_enc, self.strokes_input_fc, self.tokens_embedding, self.transformer, self.vocab_out_fc]
        for model in self.models:
            model.cuda()

        # init transformer
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Optimizers
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

    #
    # Data
    #
    def get_data_loaders(self):
        tr_dataset = ProgressionPairDataset('train')
        val_dataset = ProgressionPairDataset('valid')
        tr_loader = DataLoader(tr_dataset, batch_size=self.hp.batch_size, shuffle=True,
                               collate_fn=ProgressionPairDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.hp.batch_size, shuffle=False,
                                collate_fn=ProgressionPairDataset.collate_fn)
        return tr_loader, val_loader

    def preprocess_batch_from_data_loader(self, batch):
        """Convert tensors to cuda"""
        preprocessed = []
        for item in batch:
            if type(item) == torch.Tensor:
                item = utils.move_to_cuda(item)
            preprocessed.append(item)
        return preprocessed

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        :param batch:
            strokes: [bsz, max_stroke_len, 5] FloatTensor
            stroke_lens: list of ints
            pre_strokes: [bsz, max_pre_stroke_len, 5] FloatTensor
            pre_stroke_lens: list of ints
            texts: list of strs
            text_lens: list of ints
            text_indices_w_sos_eos: [bsz, max_text_len + 2] LongTensor (+2 for SOS and EOS)
            cats: list of strs (categories)
            cats_idx: list of ints
        
        :return: dict: 'loss': float Tensor must exist
        """
        strokes, stroke_lens, pre_strokes, pre_stroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch
        bsz = strokes.size(0)

        # Embed strokes and text
        strokes_emb = self.strokes_input_fc(strokes)                   # [bsz, max_stroke_len, dim]
        texts_emb = self.tokens_embedding(text_indices_w_sos_eos)      # [bsz, max_text_len + 2, dim]

        #
        # Encode decode with transformer
        #
        # Scaling and positional encoding
        enc_inputs = scale_add_pos_emb(strokes_emb, self.pos_enc)
        dec_inputs = scale_add_pos_emb(texts_emb, self.pos_enc)

        # transpose because transformer expects length dimension first
        enc_inputs.transpose_(0, 1)  # [max_stroke_len, bsz, dim]
        dec_inputs.transpose_(0, 1)  # [max_text_len + 2, bsz, dim]

        src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = \
            create_transformer_padding_masks(stroke_lens, text_lens)
        tgt_mask = generate_square_subsequent_mask(dec_inputs.size(0))  # [max_text_len + 2, max_text_len + 2]
        dec_outputs = self.transformer(enc_inputs, dec_inputs, # [max_text_len + 2, bsz, dim]
                                       src_key_padding_mask=src_key_padding_mask,
                                       # tgt_key_padding_mask=tgt_key_padding_mask, #  TODO: why does adding this result in Nans?
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       tgt_mask=tgt_mask
                                       )
        # src_mask and memory_mask: don't think there should be one
        if (dec_outputs != dec_outputs).any():
            import pdb; pdb.set_trace()
            return {'loss': dec_inputs.sum()}  # dummy for now

        #
        # Compute logits and loss
        #
        logits = self.vocab_out_fc(dec_outputs)  # [max_text_len + 2, bsz, vocab]

        logits = logits.transpose(0,1)  # [bsz, max_text_len + 2, vocab]
        logits = logits[:,:-1,:]  # [bsz, max_text_len + 1, vocab]; Last input is EOS, output would be EOS -> <token>. Should be ignored.
        vocab_size = logits.size(-1)
        text_indices_w_eos = text_indices_w_sos_eos[:,1:]  # remove sos; [bsz, max_text_len + 1]

        loss = F.cross_entropy(logits.reshape(-1, vocab_size),  # [bsz * max_text_len + 1, vocab]
                               text_indices_w_eos.reshape(-1),  # [bsz * max_text_len + 1]
                               ignore_index=PAD_ID)  # TODO: is ignore_index enough for masking loss value?

        result = {'loss': loss}

        return result

    def end_of_epoch_hook(self, epoch, outputs_path=None, writer=None):

        for model in self.models:
            model.eval()

        with torch.no_grad():

            # Generate texts on validation set
            generated = []
            for i, batch in enumerate(self.val_loader):
                batch = self.preprocess_batch_from_data_loader(batch)
                strokes, stroke_lens, pre_strokes, pre_stroke_lens, \
                    texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch
                bsz = strokes.size(0)

                strokes_emb = self.strokes_input_fc(strokes)                    # [bsz, max_stroke_len, dim]
                input_embs = scale_add_pos_emb(strokes_emb, self.pos_enc)    # [bsz, max_stroke_len, dim]

                init_ids = utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
                init_embs = self.tokens_embedding(init_ids)  # [bsz, 1, dim]
                decoded_probs, decoded_ids, decoded_texts = generate(
                    # TODO: probably should just pass in strokes_input_fc as well..
                    self.transformer, self.vocab_out_fc, self.tokens_embedding, self.pos_enc,
                    input_embs=input_embs, input_lens=stroke_lens,
                    init_ids=init_ids, init_embs=init_embs,
                    PAD_ID=PAD_ID, EOS_ID=EOS_ID,
                    max_len=100,
                    decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
                    idx2token=self.tr_loader.dataset.idx2token)

                for j, instruction in enumerate(texts):
                    generated.append({
                        'ground_truth': instruction,
                        'generated': decoded_texts[j],
                        'url': urls[j]
                    })
                    text = 'Ground truth: {}  \n  \nGenerated: {}  \n  \nURL: {}'.format(
                        instruction, decoded_texts[j], urls[j])
                    writer.add_text('inference/sample', text, epoch * self.val_loader.__len__() + j)

            out_fp = os.path.join(outputs_path, 'samples_e{}.json'.format(epoch))
            utils.save_file(generated, out_fp, verbose=True)


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    # Add additional arguments to parser
    opt = parser.parse_args()
    utils.setup_seeds()

    save_dir = os.path.join(RUNS_PATH, 'instructionrnn', run_name)
    utils.save_run_data(save_dir, hp)

    model = InstructionRNN(hp, save_dir)
    model.train_loop()

    # val_dataset = ProgressionPairDataset('valid')
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
    #                         collate_fn=ProgressionPairDataset.collate_fn)
    # idx2token = val_loader.dataset.idx2token
    # for batch in val_loader:
    #     strokes, stroke_lens, pre_strokes, pre_stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch
    #     import pdb; pdb.set_trace()
