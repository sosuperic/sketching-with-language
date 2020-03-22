# sketch_rnn_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname testsweep
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname categories
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname reproduce
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname drawings
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname nolayernorm_2kper
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname vae_pig
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname vae_all70k
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname gmmln_allvaryk
"""

import argparse
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_rnn.py'

NGPUS_PER_RUN = 1

GRID_REPRODUCE = {  # try to approximately reproduce results of sketchrnn paper
    'dataset': [
        # 'ndjson --max_per_category 70000',
        'npz'
    ],
    'categories': [
        'pig',
        # 'cat',
    ],
    'enc_dim': [512],
    'dec_dim': [2048],
    'enc_num_layers': [1],
    'use_categories_dec': [False],
    'model_type': [
        'decodergmm --lr 0.0001 --use_layer_norm true --dropout 0.1 --rec_dropout 0.1',
        'decodergmm --lr 0.0005 --use_layer_norm true --dropout 0.1 --rec_dropout 0.1',
        'decodergmm --lr 0.001 --use_layer_norm true --dropout 0.1 --rec_dropout 0.1',
        'decodergmm --lr 0.001 --use_layer_norm true --dropout 0.2 --rec_dropout 0.2',
    ],
}

GRID_DRAW = {
    'dataset': ['ndjson'],
    'max_per_category': [
        # '70000 --categories pig',
        # '2000 --categories all',
        # '20000 --categories all',
        '70000 --categories all',
    ],
    'enc_dim': [512],
    'dec_dim': [2048],
    'enc_num_layers': [1],
    'use_categories_dec': [True],
    'model_type': [
        # 'decodergmm --use_layer_norm true --dropout 0.1 --rec_dropout 0.1',
        'vae --use_layer_norm true --dropout 0.1 --rec_dropout 0.1'
    ],
    'lr': [0.0001],
    'batch_size': [64],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='sketchrnn',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.groupname}'

    # grid = GRID_REPRODUCE
    grid = GRID_DRAW

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    free_gpu_max_mem=0.5,
                    email_groupname=args.email_groupname)
