# strokes_to_instruction.py

"""
Use the annotated MTurk data (ProgressionPairDataset) to train a P(instruction | drawing_segment) model.

Usage:
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python src/models/strokes_to_instruction.py --model_type cnn_lstm
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python src/models/strokes_to_instruction.py --model_type cnn_lstm --use_mem true
"""

from datetime import datetime
import os

import matplotlib
matplotlib.use('Agg')
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import RUNS_PATH, \
    BEST_STROKES_TO_INSTRUCTION_PATH, \
    INSTRUCTIONS_VOCAB_DISTRIBUTION_PATH
from src.models.base.instruction_models import ProgressionPairDataset, InstructionDecoderLSTM, \
    PAD_ID, OOV_ID, SOS_ID, EOS_ID, DrawingsAsImagesAnnotatedDataset
from src.models.base.stroke_models import StrokeEncoderTransformer, StrokeEncoderLSTM, StrokeEncoderCNN, \
    StrokeAsImageEncoderCNN
from src.models.base.memory import SketchMem
from src.models.core.train_nn import TrainNN
from src.models.core.transformer_utils import *
from src.models.core import experiments, nn_utils
from src.eval.strokes_to_instruction import InstructionScorer

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
        self.max_epochs = 1000
        self.unlikelihood_loss = False

        # Dataset (and model)
        self.drawing_type = 'stroke'  # 'stroke' or 'image'
        self.cnn_type = 'wideresnet'  #'wideresnet,se,cbam' (when drawing_type == 'image')
        self.use_prestrokes = False  # for 'stroke'
        self.images = 'pre,start_to_annotated,full'  # for image; annotated,pre,post,start_to_annotated,full
        self.data_aug_on_text = True   # only for drawing_type=image right now

        # Model
        self.dim = 256
        self.n_enc_layers = 4
        self.n_dec_layers = 4
        self.model_type = 'lstm'  # 'lstm', 'transformer_lstm', 'cnn_lstm'
        self.use_layer_norm = False   # currently only for lstm
        self.condition_on_hc = False  # input to decoder also contains last hidden cell
        self.use_categories_enc = False
        self.use_categories_dec = True
        self.dropout = 0.2

        # memory
        self.use_mem = False
        self.base_mem_size = 128
        self.category_mem_size = 32
        self.mem_dim = 256

        # Additional ranking metric loss
        self.rank_imgs_text = False
        self.n_rank_imgs = 4
        self.rank_sim = 'bilinear'  # 'dot'

        # inference
        self.decode_method = 'greedy'  # 'sample', 'greedy'
        self.tau = 1.0  # sampling text
        self.k = 5      # sampling text

        # Other
        self.notes = ''


class StrokesToInstructionModel(TrainNN):
    def __init__(self, hp, save_dir=None):
        super().__init__(hp, save_dir)
        self.tr_loader = self.get_data_loader('train', shuffle=True)
        self.val_loader = self.get_data_loader('valid', shuffle=False)
        self.end_epoch_loader = self.val_loader

        #
        # Model
        #
        self.token_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.dim)
        self.models.append(self.token_embedding)
        self.category_embedding = None
        if (self.hp.use_categories_enc) or (hp.use_categories_dec):
            self.category_embedding = nn.Embedding(35, hp.dim)
            self.models.append(self.category_embedding)
        if self.hp.rank_imgs_text:
            self.rank_bilin_mod = torch.nn.Bilinear(hp.dim, hp.dim, 1)
            self.models.append(self.rank_bilin_mod)

        # Encoder decoder
        if hp.model_type.endswith('lstm'):
            if hp.drawing_type == 'image':
                self.n_channels = len(hp.images.split(','))
                self.enc = StrokeAsImageEncoderCNN(hp.cnn_type, self.n_channels, hp.dim)
            else:  # drawing_type is stroke

                # encoders may be different
                if hp.model_type == 'cnn_lstm':
                    self.enc = StrokeEncoderCNN(n_feat_maps=hp.dim, input_dim=5, emb_dim=hp.dim, dropout=hp.dropout,
                                                use_categories=hp.use_categories_enc)
                    # raise NotImplementedError('use_categories_enc=true not implemented for CNN encoder')
                elif hp.model_type == 'transformer_lstm':
                    self.enc = StrokeEncoderTransformer(
                        5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout,
                        use_categories=hp.use_categories_enc,
                    )
                elif hp.model_type == 'lstm':
                    self.enc = StrokeEncoderLSTM(
                        5, hp.dim, num_layers=hp.n_enc_layers, dropout=hp.dropout, batch_first=False,
                        use_categories=hp.use_categories_enc, use_layer_norm=hp.use_layer_norm
                    )

            if hp.use_mem:
                self.mem = SketchMem(base_mem_size=hp.base_mem_size, category_mem_size=hp.category_mem_size,
                                     mem_dim=hp.mem_dim, input_dim=hp.dim, output_dim=hp.dim)
                self.models.append(self.mem)

            # decoder is lstm
            dec_input_dim = hp.dim
            if hp.condition_on_hc:
                dec_input_dim += hp.dim
            if hp.use_categories_dec:
                dec_input_dim += hp.dim
            self.dec = InstructionDecoderLSTM(
                dec_input_dim, hp.dim, num_layers=hp.n_dec_layers, dropout=hp.dropout, batch_first=False,
                condition_on_hc=hp.condition_on_hc, use_categories=hp.use_categories_dec
            )

            self.models.extend([self.enc, self.dec])
        elif hp.model_type == 'transformer':
            if hp.use_categories_enc or hp.use_categories_dec:
                raise NotImplementedError('Use categories not implemented for Transformer')

            self.strokes_input_fc = nn.Linear(5, hp.dim)
            self.pos_enc = PositionalEncoder(hp.dim, max_seq_len=250)
            self.transformer = nn.Transformer(
                d_model=hp.dim, dim_feedforward=hp.dim * 4, nhead=2, activation='relu',
                num_encoder_layers=hp.n_enc_layers, num_decoder_layers=hp.n_dec_layers,
                dropout=hp.dropout,
            )
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            self.models.extend([self.strokes_input_fc, self.pos_enc, self.transformer])

        for model in self.models:
            model.cuda()

        # Additional loss
        if hp.unlikelihood_loss:
            # assert hp.model_type == 'cnn_lstm'

            # load true vocab distribution
            token2idx = self.tr_loader.dataset.token2idx
            vocab_prob = utils.load_file(INSTRUCTIONS_VOCAB_DISTRIBUTION_PATH)
            self.vocab_prob = torch.zeros(len(token2idx)).fill_(1e-6)  # fill with eps
            for token, prob in vocab_prob.items():
                try:
                    idx = token2idx[token]
                    self.vocab_prob[idx] = prob
                except KeyError as e:  # not sure why 'lion' isn't in vocab
                    print(e)
                    continue
            self.vocab_prob = nn_utils.move_to_cuda(self.vocab_prob)  # [vocab]

            # create running of vocab distribution
            n_past = 256  # number of minibatches to store
            self.model_vocab_prob = torch.zeros(n_past, len(token2idx))  # [n, vocab]
            self.model_vocab_prob = nn_utils.move_to_cuda(self.model_vocab_prob)


        # Optimizers
        self.optimizers.append(optim.Adam(self.parameters(), hp.lr))

        self.scorers = [InstructionScorer('rouge')]

    def load_enc_weights_from_autoencoder(self, dir):
        """
        Copy pretrained encoder weights to self.encoder.

        Args:
            dir: str
        """
        print('Loading encoder weights from pretrained autoencoder: ', dir)
        fp = os.path.join(dir, 'model.pt')
        trained_state_dict = torch.load(fp)
        self_state_dict = self.state_dict()
        for name, param in trained_state_dict.items():
            if name.startswith('enc'):
                self_state_dict[name].copy_(param)

    #
    # Data
    #
    def get_data_loader(self, dataset_split, shuffle=False):
        """
        Args:
            dataset_split (str): 'train', 'valid', 'test'
            shuffle (bool)
        """
        if self.hp.drawing_type == 'stroke':
            ds = ProgressionPairDataset(dataset_split, use_prestrokes=self.hp.use_prestrokes)
            loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=shuffle,
                                collate_fn=ProgressionPairDataset.collate_fn)
        elif self.hp.drawing_type == 'image':
            ds = DrawingsAsImagesAnnotatedDataset(dataset_split, images=self.hp.images,
                                                  data_aug_on_text=self.hp.data_aug_on_text,
                                                  rank_imgs_text=self.hp.rank_imgs_text,
                                                  n_rank_imgs=self.hp.n_rank_imgs)
            loader = DataLoader(ds, batch_size=self.hp.batch_size, shuffle=shuffle,
                                collate_fn=DrawingsAsImagesAnnotatedDataset.collate_fn)
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

    def compute_loss(self, logits, tf_inputs, pad_id):
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
        loss = F.cross_entropy(logits, targets, ignore_index=pad_id)

        return loss

    def compute_unlikelihood_loss(self, logits, text_lens):
        loss = torch.tensor(0.)

        # Keep track of last n batches of probs. Shift by 1, add latest
        # updated = nn_utils.move_to_cuda(torch.zeros_like(self.model_vocab_prob))
        updated = self.model_vocab_prob.clone().detach()  # TODO: where to detach...
        updated[:updated.size(0) - 1, :] = self.model_vocab_prob[1:,:].detach()  # detach?
        self.model_vocab_prob = updated

        # Detaching above so that it doesn't backprop through all entire model_vocab_probs, just the current one?

        # compute models' vocab prob in current batch
        probs = F.softmax(logits, dim=-1)  # [len, bsz, vocab]
        logits_len, bsz, _ = logits.size()
        mask = nn_utils.move_to_cuda(torch.zeros(logits_len, bsz))  # [len, bsz]
        for i in range(bsz):
            mask[:text_lens[i]] = 1
        mask = mask.unsqueeze(-1)       # [len, bsz, vocab]

        batch_model_prob = (probs * mask).mean(dim=0).mean(dim=0)  # [vocab]
        self.model_vocab_prob[-1] = batch_model_prob

        # Only compute after having seen n batches
        if self.model_vocab_prob[0].sum() == 0:
            return loss

        cur_model_vocab = self.model_vocab_prob.mean(dim=0)  # [vocab]
        mismatch = cur_model_vocab * torch.log(cur_model_vocab / self.vocab_prob.detach())
        unlikelihood = torch.log(1 - probs) * mask  # [len, bsz, vocab]
        loss = -(mismatch * unlikelihood).mean()

        loss *= 500  # mixing parameter
        # print('ull, ', loss.item())

        return loss

    def rank_imgs_text_loss(self, imgs_emb, texts_emb, imgs_pref):
        """
        Compute list-wise ranking loss. Text_emb is the query,
        imgs_emb are list of possible images being ranked, imgs_pref is the
        ranking we are trying to achieve.

        Args:
            imgs_emb ([bsz, n_rank_imgs, dim])
            texts_emb ([bsz, dim])
            imgs_pref ([bsz, n_rank_imgs]): Target ranking preference

        Returns:
            loss: FloatTensor
        """
        # Compute similarity between images and text
        if self.hp.rank_sim == 'dot':
            # Bmm([b * n * m], [b * m * p]) -> [b * n * p]
            texts_emb = texts_emb.unsqueeze(2)  # [bsz, dim, 1]
            sims = imgs_emb.bmm(texts_emb)  # [bsz, n_rank_imgs, 1]
            sims = sims.squeeze()  # [bsz, n_rank_imgs]
        elif self.hp.rank_sim == 'bilinear':
            texts_emb = texts_emb.unsqueeze(1).repeat(1, self.hp.n_rank_imgs, 1)  # [bsz, n_rank_imgs, dim]
            sims = self.rank_bilin_mod(imgs_emb, texts_emb)  # [bsz, n_rank_imgs, 1]
            sims = sims.squeeze()  # [bsz, n_rank_imgs]

        # Cross entropy (-p * log q)
        loss = torch.mean(torch.sum(-imgs_pref * F.log_softmax(sims, dim=1), dim=1))

        # print(sims[0].cpu().detach().numpy().tolist())
        # print(imgs_pref[0].cpu().detach().numpy().tolist())
        # print(loss)
        return loss

    def one_forward_pass(self, batch):
        """
        Return loss and other items of interest for one forward pass

        Args: batch: tuple from DataLoader
            strokes: [max_stroke_len, bsz, 5] FloatTensor
            stroke_lens: list of ints
            texts: list of strs
            text_lens: list of ints
            text_indices_w_sos_eos: [max_text_len + 2, bsz] LongTensor (+2 for sos and eos)
            cats: list of strs (categories)
            cats_idx: list of ints

        :return: dict: 'loss': float Tensor must exist
        """
        if self.hp.drawing_type == 'image':
            return self.one_forward_pass_imagecnn_lstm(batch)
        elif self.hp.drawing_type == 'stroke':
            if self.hp.model_type == 'cnn_lstm':
                return self.one_forward_pass_cnn_lstm(batch)
            elif self.hp.model_type == 'transformer_lstm':
                return self.one_forward_pass_transformer_lstm(batch)
            elif self.hp.model_type == 'lstm':
                return self.one_forward_pass_lstm(batch)
            elif self.hp.model_type == 'transformer':
                return self.one_forward_pass_transformer(batch)


    def one_forward_pass_imagecnn_lstm(self, batch):
        imgs, (rank_imgs, rank_imgs_pref), texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        embedded = self.enc(imgs)  # [bsz, dim]

        if self.hp.use_mem:
            # mem_emb = self.mem(embedded, cats_idx)  # [bsz, mem_dim]
            embedded = embedded + self.mem(embedded, cats_idx)  # [bsz, mem_dim]
            mem_emb = None

        embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
        hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
        cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)  # [max_text_len + 2, bsz, dim]
        logits, texts_hidden = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                                        token_embedding=self.token_embedding,
                                        category_embedding=self.category_embedding, categories=cats_idx,  # [max_text_len + 2, bsz, vocab]; h/c
                                        mem_emb=mem_emb)
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss, 'loss_decode': loss.clone().detach()}

        if self.hp.unlikelihood_loss:
            loss_UL = self.compute_unlikelihood_loss(logits, text_lens)
            result['loss'] += loss_UL  # for backward
            result['loss_unlikelihood'] = loss_UL.clone().detach()  # for logging

        if self.hp.rank_imgs_text:
            # not on cuda because it's a tuple within the batch... not done by preprocess()
            rank_imgs = nn_utils.move_to_cuda(rank_imgs)
            rank_imgs_pref = nn_utils.move_to_cuda(rank_imgs_pref)

            # embed rank images
            C, bsz, H, W = imgs.size()
            rank_imgs = rank_imgs.view(bsz * self.hp.n_rank_imgs, C, H, W)  # [bsz, rank_n_imgs, C, H, W] ->  [bsz * rank_n_imgs, C, H, W]
            rank_imgs = rank_imgs.transpose(0,1) # [C, bsz * rank_n_imgs, H, W]  (CNN expects batch second)
            rank_imgs_emb = self.enc(rank_imgs)   # [bsz * rank_n_imgs, dim]
            rank_imgs_emb = rank_imgs_emb.view(bsz, self.hp.n_rank_imgs, -1)  # [bsz, rank_n_imgs, dim]

            # Compute loss
            texts_hidden = texts_hidden[0][-1,:,:]  # last layer hidden? -> [bsz, dim]  # TODO:
            loss_rank = self.rank_imgs_text_loss(rank_imgs_emb, texts_hidden, rank_imgs_pref)
            result['loss'] += loss_rank  # for backward
            result['loss_rank_imgs_text'] = loss_rank.clone().detach()  # for logging

        return result

    def one_forward_pass_cnn_lstm(self, batch):
        strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        embedded = self.enc(strokes, stroke_lens,
                            category_embedding=self.category_embedding, categories=cats_idx)
        # [bsz, dim]
        if self.hp.use_mem:
            embedded = embedded + self.mem(embedded, cats_idx)  # [bsz, dim]

        embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
        hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
        cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)  # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, vocab]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss, 'loss_decode': loss.clone().detach()}

        if self.hp.unlikelihood_loss:
            loss_UL = self.compute_unlikelihood_loss(logits, text_lens)

            result['loss'] += loss_UL  # for backward
            result['loss_unlikelihood'] = loss_UL.clone().detach()  # for logging

        return result

    def one_forward_pass_transformer_lstm(self, batch):
        strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        hidden = self.enc(strokes, stroke_lens,
                          category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, dim]
        # [bsz, dim]
        hidden = hidden.unsqueeze(0)  # [1, bsz, dim]
        hidden = hidden.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
        cell = hidden.clone()  # [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)  # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, vocab]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss, 'loss_decode': loss.clone().detach()}

        return result

    def one_forward_pass_lstm(self, batch):
        strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        _, (hidden, cell) = self.enc(strokes, stroke_lens,
                                     category_embedding=self.category_embedding, categories=cats_idx)
        # [bsz, max_stroke_len, dim]; h/c = [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)      # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, vocab]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss, 'loss_decode': loss.clone().detach()}

        return result

    def one_forward_pass_transformer(self, batch):
        strokes, stroke_lens, texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Embed strokes and text
        strokes_emb = self.strokes_input_fc(strokes)                   # [max_stroke_len, bsz, dim]
        texts_emb = self.token_embedding(text_indices_w_sos_eos)      # [max_text_len + 2, bsz, dim]

        #
        # Encode decode with transformer
        #
        # Scaling and positional encoding
        enc_inputs = scale_add_pos_emb(strokes_emb, self.pos_enc)
        dec_inputs = scale_add_pos_emb(texts_emb, self.pos_enc)

        src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = \
            create_transformer_padding_masks(stroke_lens, text_lens)
        tgt_mask = generate_square_subsequent_mask(dec_inputs.size(0))  # [max_text_len + 2, max_text_len + 2]
        dec_outputs = self.transformer(enc_inputs, dec_inputs,
                                       src_key_padding_mask=src_key_padding_mask,
                                       # tgt_key_padding_mask=tgt_key_padding_mask, #  TODO: why does adding this result in Nans?
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       tgt_mask=tgt_mask)
        # dec_outputs: [max_text_len + 2, bsz, dim]

        if (dec_outputs != dec_outputs).any():
            import pdb; pdb.set_trace()

        # Compute logits and loss
        logits = torch.matmul(dec_outputs, self.token_embedding.weight.t())  # [max_text_len + 2, bsz, vocab]
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss, 'loss_decode': loss.clone().detach(), 'logits': logits}

        return result


    # End of epoch hook
    def end_of_epoch_hook(self, data_loader, epoch, outputs_path=None, writer=None):
        """
        Args:
            data_loader: DataLoader
            epoch: int
            outputs_path: str
            writer: Tensorboard Writer
        """
        for model in self.models:
            model.eval()

        with torch.no_grad():
            # Generate texts on validation set
            inference = self.inference_loop(data_loader, writer=writer, epoch=epoch)
            out_fp = os.path.join(outputs_path, 'samples_e{}.json'.format(epoch))
            utils.save_file(inference, out_fp, verbose=True)

    def inference_pass(self, strokes, stroke_lens, cats_idx):
        """

        Args:
            strokes: [len, bsz, 5]
            stroke_lens: list of ints
            cats_idx: [bsz] LongTensor

        Returns:
            decoded_probs: [bsz, max_len, vocab]
            decoded_ids: [bsz, max_len]
            decoded_texts: list of strs
        """
        bsz = strokes.size(1)

        if self.hp.model_type in ['cnn_lstm', 'transformer_lstm', 'lstm']:
            mem_emb = None
            if self.hp.drawing_type == 'image':
                # TODO: this is horribly confusing...
                # strokes is actually images [C, B, H, W]
                # stroke_lens (2nd item in batch) is actually a tuple of rank_imgs and rank_imgs_pref
                # We don't need to use rank imgs during inference, it's just used during training as an auxiliary loss
                embedded = self.enc(strokes)  # [bsz, dim]
                if self.hp.use_mem:
                    # mem_emb = self.mem(embedded, cats_idx)  # [bsz, mem_dim]
                    embedded = embedded + self.mem(embedded, cats_idx)  # [bsz, mem_dim]

                embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
                hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_  glayers, bsz, dim]
                cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
            else:
                if self.hp.model_type == 'cnn_lstm':
                    # Encode strokes
                    embedded = self.enc(strokes, stroke_lens,
                                        category_embedding=self.category_embedding, categories=cats_idx)
                    # [bsz, dim]
                    if self.hp.use_mem:
                        # mem_emb = self.mem(embedded, cats_idx)  # [bsz, mem_dim]
                        embedded = embedded + self.mem(embedded, cats_idx)  # [bsz, mem_dim]
                    embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
                    hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
                    cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
                elif self.hp.model_type == 'transformer_lstm':
                    # Encode strokes
                    hidden = self.enc(strokes, stroke_lens,
                                    category_embedding=self.category_embedding, categories=cats_idx)  # [bsz, dim]
                    # [bsz, dim]
                    hidden = hidden.unsqueeze(0)  # [1, bsz, dim]
                    hidden = hidden.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
                    cell = hidden.clone()  # [n_layers, bsz, dim]

                elif self.hp.model_type == 'lstm':
                    _, (hidden, cell) = self.enc(strokes, stroke_lens,
                                                category_embedding=self.category_embedding, categories=cats_idx)
                    # [max_stroke_len, bsz, dim]; h/c = [layers * direc, bsz, dim]

            # Create init input
            init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
            init_ids.transpose_(0, 1)  # [1, bsz]

            decoded_probs, decoded_ids, decoded_texts = self.dec.generate(
                self.token_embedding,
                category_embedding=self.category_embedding, categories=cats_idx,
                mem_emb=mem_emb,
                init_ids=init_ids, hidden=hidden, cell=cell,
                pad_id=PAD_ID, eos_id=EOS_ID, max_len=25,
                decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
                idx2token=self.tr_loader.dataset.idx2token,
            )

        elif self.hp.model_type == 'transformer':
            strokes_emb = self.strokes_input_fc(strokes)  # [max_stroke_len, bsz, dim]
            src_input_embs = scale_add_pos_emb(strokes_emb, self.pos_enc)  # [max_stroke_len, bsz, dim]

            init_ids = nn_utils.move_to_cuda(torch.LongTensor([SOS_ID] * bsz).unsqueeze(1))  # [bsz, 1]
            init_ids.transpose_(0, 1)  # [1, bsz]
            init_embs = self.token_embedding(init_ids)  # [1, bsz, dim]

            decoded_probs, decoded_ids, decoded_texts = transformer_generate(
                self.transformer, self.token_embedding, self.pos_enc,
                src_input_embs=src_input_embs, input_lens=stroke_lens,
                init_ids=init_ids,
                pad_id=PAD_ID, eos_id=EOS_ID,
                max_len=100,
                decode_method=self.hp.decode_method, tau=self.hp.tau, k=self.hp.k,
                idx2token=self.tr_loader.dataset.idx2token)

        return decoded_probs, decoded_ids, decoded_texts

    def inference_loop(self, loader, writer=None, epoch=None):
        """
        Args:
            loader: DataLoader
            writer: Tensorboard writer (used during validation)
            epoch: int (used for writer)

        Returns: list of dicts
        """
        inference = []
        for i, batch in enumerate(loader):
            batch = self.preprocess_batch_from_data_loader(batch)
            strokes, stroke_lens, texts, _, _, cats, cats_idx, urls = batch
            decoded_probs, decoded_ids, decoded_texts = self.inference_pass(strokes, stroke_lens, cats_idx)

            for j, instruction in enumerate(texts):
                gt, gen = instruction, decoded_texts[j]

                # construct results
                result = {
                    'ground_truth': gt,
                    'generated': gen,
                    'url': urls[j],
                    'category': cats[j],
                }
                for scorer in self.scorers:
                    for name, value in scorer.score(gt, gen).items():
                        result[name] = value
                        if writer:
                            writer.add_scalar('inference/{}'.format(name), value, epoch * self.val_loader.__len__() + i)
                inference.append(result)

                # log
                text = 'Ground truth: {}  \n  \nGenerated: {}  \n  \nURL: {}'.format(
                    instruction, decoded_texts[j], urls[j])
                if writer:
                    writer.add_text('inference/sample', text, epoch * self.val_loader.__len__() + i)

        return inference


if __name__ == '__main__':
    hp = HParams()
    hp, run_name, parser = experiments.create_argparse_and_update_hp(hp)
    parser.add_argument('--groupname', default='debug', help='name of subdir to save runs')
    parser.add_argument('--load_autoencoder_dir', default=None, help='directory that contains pretrained autoencoder'
                        'from which we can load the encoder weights')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_split', default='valid', help='dataset split to perform inference on')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    if opt.inference:
        # TODO: should load hp before... write function in utils
        model = StrokesToInstructionModel(hp, save_dir=None)
        model.load_model(BEST_STROKES_TO_INSTRUCTION_PATH)
        model.save_inference_on_split(dataset_split=opt.inference_split,
                                      dir=BEST_STROKES_TO_INSTRUCTION_PATH, ext='json')

    else:
        save_dir = os.path.join(RUNS_PATH, 'strokes_to_instruction', datetime.today().strftime('%b%d_%Y'), opt.groupname, run_name)
        model = StrokesToInstructionModel(hp, save_dir)
        experiments.save_run_data(save_dir, hp)

        if opt.load_autoencoder_dir:
            model.load_enc_weights_from_autoencoder(opt.load_autoencoder_dir)

        model.train_loop()



    # Testing / debugging data
    # val_dataset = ProgressionPairDataset('valid')
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
    #                         collate_fn=ProgressionPairDataset.collate_fn)
    # idx2token = val_loader.dataset.idx2token
    # for batch in val_loader:
    #     strokes, stroke_lens, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch
    #     import pdb; pdb.set_trace()
