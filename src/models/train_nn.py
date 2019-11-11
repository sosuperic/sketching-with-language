# train_nn.py

from collections import defaultdict
from datetime import datetime
import numpy as np
import os
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

RUNS_PATH = 'runs/'

##############################################################################
#
# MODEL
#
##############################################################################

class TrainNN(nn.Module):

    """
    Base class for this project that covers train-eval loop.
    """

    def __init__(self, hp, save_dir):
        super().__init__()

        self.hp = hp
        self.save_dir = save_dir

        self.models = []
        self.optimizers = []

        self.tr_loader = None
        self.val_loader = None

    def get_data_loader(self):
        pass

    #
    # Training
    #
    def lr_decay(self, optimizer, min_lr, lr_decay):
        """
        Decay learning rate by a factor of lr_decay
        """
        for param_group in optimizer.param_groups:
            if param_group['lr'] > min_lr:
                param_group['lr'] *= lr_decay
        return optimizer

    def preprocess_batch_from_data_loader(self, batch):
        return batch

    def one_forward_pass(self, batch):
        """
        Return dict where values are float Tensors. Key 'loss' must exist.
        """
        pass

    def pre_forward_train_hook(self):
        """Called before one forward pass when training"""
        pass

    def dataset_loop(self, data_loader, epoch, is_train=True, writer=None, tb_tag='train'):
        """
        Can be used with either different splits of the dataset (train, valid, test)

        :return: dict
            key: str
            value: float
        """
        losses = defaultdict(list)
        for i, batch in enumerate(data_loader):
            # set up optimizers
            if is_train:
                self.pre_forward_train_hook()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

            # forward pass
            batch = self.preprocess_batch_from_data_loader(batch)  # [max_len, bsz, 5];
            result = self.one_forward_pass(batch)
            for k, v in result.items():
                if k.startswith('loss'):
                    losses[k].append(v.item())

            # optimization
            if is_train:
                result['loss'].backward()
                nn.utils.clip_grad_value_(self.parameters(), self.hp.grad_clip)
                for optimizer in self.optimizers:
                    optimizer.step()

            # Logging
            if i % 10 == 0:
                step = epoch * data_loader.__len__() + i
                for k, v in result.items():
                    if k.startswith('loss'):
                        writer.add_scalar('{}/{}'.format(tb_tag, k), v.item(), step)

        mean_losses = {k: np.mean(values) for k, values in losses.items()}
        return mean_losses

    def get_log_str(self, epoch, dataset_split, mean_losses, runtime=None):
        """
        Create string to log to stdout
        """
        log_str = 'Epoch {} -- {}:'.format(epoch, dataset_split)
        for k, v in mean_losses.items():
            log_str += ' {}={:.4f}'.format(k, v)
        if runtime is not None:
            log_str += ' minutes={:.1f}'.format(runtime)
        return log_str

    def train_loop(self):
        """Train and validate on multiple epochs"""
        tb_path = os.path.join(self.save_dir, 'tensorboard')
        outputs_path = os.path.join(self.save_dir, 'outputs')
        os.makedirs(outputs_path)
        writer = SummaryWriter(tb_path)
        stdout_fp = os.path.join(self.save_dir, 'stdout.txt')
        stdout_f = open(stdout_fp, 'w')

        # Train
        val_losses = []  # used for early stopping
        min_val_loss = float('inf')  # used to save model
        for epoch in range(self.hp.max_epochs):

            # train
            start_time = time.time()
            for model in self.models:
                model.train()
            mean_losses = self.dataset_loop(self.tr_loader, epoch, is_train=True, writer=writer, tb_tag='train')
            end_time = time.time()
            min_elapsed = (end_time - start_time) / 60
            log_str = self.get_log_str(epoch, 'train', mean_losses, runtime=min_elapsed)
            print(log_str, file=stdout_f)
            stdout_f.flush()

            for optimizer in self.optimizers:
                self.lr_decay(optimizer, self.hp.min_lr, self.hp.lr_decay)

            # validate
            for model in self.models:
                model.eval()
            mean_losses = self.dataset_loop(self.val_loader, epoch, is_train=False, writer=writer, tb_tag='valid')
            val_loss = mean_losses['loss']
            val_losses.append(val_loss)
            # TODO: early stopping
            log_str = self.get_log_str(epoch, 'valid', mean_losses)
            print(log_str, file=stdout_f)
            stdout_f.flush()

            # Generate sequence to save image and show progress
            self.end_of_epoch_hook(epoch, outputs_path=outputs_path)

            # Save model. TODO: only save best model (model.pt)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                model_fn = 'e{}_loss{:.4f}.pt'.format(epoch, val_loss)  # val loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, model_fn))

        stdout_f.close()

    def end_of_epoch_hook(self, epoch, outputs_path=None):  # TODO: is this how to use **kwargs
        pass
