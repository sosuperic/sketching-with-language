# utils.py

import argparse
from datetime import datetime
import json
import os
import pickle
import shutil
import sys
import torch
import torch.nn.functional as F
import uuid


#################################################
#
# Simple I/O
#
#################################################

def save_file(data, path, verbose=False):
    """Creates intermediate directories if they don't exist."""
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if verbose:
        print('Saving: {}'.format(path))

    _, ext = os.path.splitext(path)
    if ext == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=2)
    elif ext == '.json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')  # add trailing newline for POSIX compatibility

def load_file(path):
    _, ext = os.path.splitext(path)
    if ext == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    return data

#################################################
#
# Neural Network
#
#################################################

def move_to_cuda(x):
    """Move tensor to cuda"""
    if torch.cuda.is_available():
        if type(x) == tuple:
            x = tuple([t.cuda() for t in x])
        else:
            x = x.cuda()
    return x

def setup_seeds(seed=1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

#
# Decoding utils
#
def logits_to_prob(logits, method='softmax',
                   tau=1.0, eps=1e-10, gumbel_hard=False):
    """
    Args:
        logits: [batch_size, vocab_size]
        method: 'gumbel', 'softmax'
        gumbel_hard: boolean
        topk: int (used for beam search)
    Returns: [batch_size, vocab_size]
    """
    if tau == 0.0:
        raise ValueError(
            'Temperature should not be 0. If you want greedy decoding, pass "greedy" to prob_to_vocab_id()')
    if method == 'gumbel':
        prob = F.gumbel_softmax(logits, tau=tau, eps=eps, hard=gumbel_hard)
    elif method == 'softmax':
        prob = F.softmax(logits / tau, dim=1)
    return prob

def prob_to_vocab_id(prob, method, k=10):
    """
    Produce vocab id given probability distribution over vocab
    Args:
        prob: [batch_size, vocab_size]
        method: str ('greedy', 'sample')
        k: int (used for beam search, sampling)
    Returns:
        prob: [batch_size * k, vocab_size]
            Rows are repeated:
                [[0.3, 0.2, 0.5],
                 [0.1, 0.7, 0.2]]
            Becomes (with k=2):
                [[0.3, 0.2, 0.5],
                 [0.3, 0.2, 0.5],
                 [0.1, 0.7, 0.2]
                 [0.1, 0.7, 0.2]]
        ids: [batch_size, k] LongTensor
    """
    if method == 'greedy':
        _, ids = torch.topk(prob, 1, dim=1)
    elif method == 'sample':
        ids = torch.multinomial(prob, k)
    return prob, ids


#################################################
#
# Hyperparams and saving experiments data
#
#################################################

def save_run_data(path_to_dir, hp):
    """
    1) Save stdout to file
    2) Save files to path_to_dir:
        - code_snapshot/: Snapshot of code (.py files)
        - hp.json: dict of HParams object
        - run_details.txt: command used and start time
    """
    print('Saving run data to: {}'.format(path_to_dir))
    parent_dir = os.path.dirname(path_to_dir)
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.isdir(path_to_dir):
        print('Data already exists in this directory (presumably from a previous run)')
        inp = input('Enter "y" if you are sure you want to remove all the old contents: ')
        if inp == 'y':
            print('Removing old contents')
            shutil.rmtree(path_to_dir)
        else:
            print('Exiting')
            raise SystemExit
    print('Creating directory and saving data')
    os.mkdir(path_to_dir)

    # Save snapshot of code
    snapshot_dir = os.path.join(path_to_dir, 'code_snapshot')
    print('Saving code snapshot to: {}'.format(snapshot_dir))
    if os.path.exists(snapshot_dir):  # shutil doesn't work if dest already exists
        shutil.rmtree(snapshot_dir)
    shutil.copytree('src', snapshot_dir)

    # Save hyperparms
    save_file(vars(hp), os.path.join(path_to_dir, 'hp.json'), verbose=True)

    # Save some command used to run, start time
    with open(os.path.join(path_to_dir, 'run_details.txt'), 'w') as f:
        f.write('Command:\n')
        cmd = ' '.join(sys.argv)
        start_time = datetime.now().strftime('%B%d_%H-%M-%S')
        f.write(cmd + '\n')
        f.write('Start time: {}'.format(start_time))
        print('Command used to start program:\n', cmd)
        print('Start time: {}'.format(start_time))

def create_argparse_and_update_hp(hp):
    """
    Args:
        hp: instance of HParams object
    Returns:
        (updated) hp
        run_name: str (can be used to create directory and store training results)
        parser: argparse object (can be used to add more arguments)
    """

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Create argparse with an option for every param in hp
    parser = argparse.ArgumentParser()
    for param, default_value in vars(hp).items():
        param_type = type(default_value)
        param_type = str2bool if param_type == bool else param_type
        parser.add_argument('--{}'.format(param), dest=param, default=None, type=param_type)
    opt, unknown = parser.parse_known_args()

    # Update hp if any command line arguments passed
    # Also create description of run
    run_name = []
    for param, value in vars(opt).items():
        if value is not None:
            setattr(hp, param, value)
            if param != 'model_type':
                run_name.append('{}_{}'.format(param, value))
    run_name = '-'.join(sorted(run_name))
    run_name = ('default_' + str(uuid.uuid4())[:8]) if (run_name == '') else run_name

    return hp, run_name, parser