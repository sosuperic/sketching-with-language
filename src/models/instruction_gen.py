# instruction_gen.py

import matplotlib
matplotlib.use('Agg')
import os

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.models.core.train_nn import TrainNN, RUNS_PATH
from src.models.core.transformer_utils import *
from src.models.base.instruction_models import ProgressionPairDataset, InstructionDecoderLSTM, \
    PAD_ID, OOV_ID, SOS_ID, EOS_ID
from src.models.base.stroke_models import StrokeEncoderTransformer, StrokeEncoderLSTM, StrokeEncoderCNN
from src.eval.stroke_to_instruction import InstructionScorer

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

        # Model
        self.dim = 512
        self.n_enc_layers = 4
        self.n_dec_layers = 4
        self.model_type = 'cnn_lstm'  # 'lstm', 'transformer_lstm', 'cnn_lstm'
        self.condition_on_hc = False  # input to decoder also contains last hidden cell
        self.use_prestrokes = True
        self.use_categories_enc = False
        self.use_categories_dec = True
        self.dropout = 0.2

        # inference
        self.decode_method = 'greedy'  # 'sample', 'greedy'
        self.tau = 1.0  # sampling text
        self.k = 5      # sampling text

        # Other
        self.notes = ''


class StrokeToInstructionModel(TrainNN):
    def __init__(self, hp, save_dir=None):
        super().__init__(hp, save_dir)

        self.tr_loader =  self.get_data_loader('train', self.hp.batch_size, shuffle=True,
                                               use_prestrokes=self.hp.use_prestrokes)
        self.val_loader = self.get_data_loader('valid', self.hp.batch_size, shuffle=False,
                                               use_prestrokes=self.hp.use_prestrokes)
        self.end_epoch_loader = self.val_loader

        # Model
        self.token_embedding = nn.Embedding(self.tr_loader.dataset.vocab_size, hp.dim)
        self.models.append(self.token_embedding)
        self.category_embedding = None
        if (self.hp.use_categories_enc) or (self.hp.use_categories_dec):
            self.category_embedding = nn.Embedding(35, self.hp.dim)
            self.models.append(self.category_embedding)

        if hp.model_type.endswith('lstm'):
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
                    use_categories=hp.use_categories_enc,
                )

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
    def get_data_loader(self, dataset_split, batch_size, shuffle=True, use_prestrokes=False):
        """
        Args:
            dataset_split: str
            batch_size: int
            shuffle: bool
        """
        ds = ProgressionPairDataset(dataset_split, use_prestrokes=use_prestrokes)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=ProgressionPairDataset.collate_fn)
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
        if self.hp.model_type == 'cnn_lstm':
            return self.one_forward_pass_cnn_lstm(batch)
        elif self.hp.model_type == 'transformer_lstm':
            return self.one_forward_pass_transformer_lstm(batch)
        elif self.hp.model_type == 'lstm':
            return self.one_forward_pass_lstm(batch)
        elif self.hp.model_type == 'transformer':
            return self.one_forward_pass_transformer(batch)

    def one_forward_pass_cnn_lstm(self, batch):
        strokes, stroke_lens, \
            texts, text_lens, text_indices_w_sos_eos, cats, cats_idx, urls = batch

        # Encode strokes
        embedded = self.enc(strokes, stroke_lens,
                            category_embedding=self.category_embedding, categories=cats_idx)
        # [bsz, dim]
        embedded = embedded.unsqueeze(0)  # [1, bsz, dim]
        hidden = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]
        cell = embedded.repeat(self.dec.num_layers, 1, 1)  # [n_layers, bsz, dim]

        # Decode
        texts_emb = self.token_embedding(text_indices_w_sos_eos)  # [max_text_len + 2, bsz, dim]
        logits, _ = self.dec(texts_emb, text_lens, hidden=hidden, cell=cell,
                             token_embedding=self.token_embedding,
                             category_embedding=self.category_embedding, categories=cats_idx)  # [max_text_len + 2, bsz, vocab]; h/c
        loss = self.compute_loss(logits, text_indices_w_sos_eos, PAD_ID)
        result = {'loss': loss}

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
        result = {'loss': loss}

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
        result = {'loss': loss}

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
        result = {'loss': loss, 'logits': logits}

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

        # Model-specific decoding
        if self.hp.model_type in ['cnn_lstm', 'transformer_lstm', 'lstm']:
            if self.hp.model_type == 'cnn_lstm':
                # Encode strokes
                embedded = self.enc(strokes, stroke_lens,
                                    category_embedding=self.category_embedding, categories=cats_idx)
                # [bsz, dim]
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
    hp, run_name, parser = utils.create_argparse_and_update_hp(hp)
    # Add additional arguments to parser
    parser.add_argument('--load_autoencoder_dir', default=None, help='directory that contains pretrained autoencoder'
                        'from which we can load the encoder weights')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_split', default='valid', help='dataset split to perform inference on')
    opt = parser.parse_args()
    nn_utils.setup_seeds()

    if opt.inference:
        BEST_STROKE_TO_INSTRUCTION_DIR = 'best_models/stroke2instruction/catsdecoder-dim_512-model_type_cnn_lstm-use_prestrokes_False/'
        # TODO: should load hp before... write function in utils
        model = StrokeToInstructionModel(hp, save_dir=None)
        model.load_model(BEST_STROKE_TO_INSTRUCTION_DIR)
        model.save_inference_on_split(dataset_split=opt.inference_split,
                                      dir=BEST_STROKE_TO_INSTRUCTION_DIR, ext='json')

    else:
        save_dir = os.path.join(RUNS_PATH, 'stroke2instruction', run_name)
        model = StrokeToInstructionModel(hp, save_dir)
        utils.save_run_data(save_dir, hp)

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
