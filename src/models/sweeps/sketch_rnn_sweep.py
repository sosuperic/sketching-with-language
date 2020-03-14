# sketch_rnn_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname testsweep
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname categories
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname reproduce
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname drawings
"""

import argparse
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_rnn.py'

NGPUS_PER_RUN = 1

GRID_REPRODUCE = {  # try to approximately reproduce results of sketchrnn paper
    'dataset': ['ndjson'],  # they use npz dataset actually
    'categories': ['pig', 'cat'],
    'max_per_category': [70000],

    'enc_dim': [512],
    'dec_dim': [2048],
    'enc_num_layers': [1],
    'use_categories_dec': [False],
    'model_type': ['decodergmm'],

    'lr': [0.0001],
}

GRID_DRAW = {
    'dataset': ['ndjson'],
    'max_per_category': [
        '70000 --categories pig',
        # '2000 --categories all',
        # '20000 --categories all',
    ],
    'enc_dim': [512],
    'dec_dim': [2048],
    'enc_num_layers': [1],
    'use_categories_dec': [True],
    'model_type': [
        # 'decodergmm',
        'decodergmm --use_layer_norm true --dropout 0.1 --rec_dropout 0.1'
    ],
    'lr': [0.0001],
}

GRID = {
    # Data
    'dataset': [
        'ndjson'
    ],
    'max_per_category': [
        # 250,
        2500,
        # 25000,
    ],
    # 'categories': [
    #     'all',
    # ],
    # Training
    'lr': [
        0.0005,
        0.0001
    ],
    # Model
    'model_type': ['decodergmm'],
    'enc_dim': [
        '256 --dec_dim 256',
        '512 --dec_dim 512',
        # '1024 --dec_dim 1024',
        # '2048 --dec_dim 2048'
    ],
    'use_categories_dec': [
        'true',
        # 'false',
    ],
    'categories_dim': [
        128,
        256,
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='sketchrnn',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    groupname = args.groupname
    if groupname is None:
        # Assumes one dataset and one max_per_category at a time right now
        groupname = f"{GRID['dataset'][0]}_{GRID['max_per_category'][0]}per"
    base_cmd = CMD + f' --groupname {groupname}'
    # print(base_cmd)

    # grid = GRID
    grid = GRID_DRAW

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    free_gpu_max_mem=0.67,
                    gpus=[6,7],
                    email_groupname=args.email_groupname)
