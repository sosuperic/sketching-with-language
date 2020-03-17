# sketch_rnn_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/retrieve_refine_sweep.py --groupname firstpass
"""

import argparse
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/retrieve_refine.py'

NGPUS_PER_RUN = 1

GRID = {
    'dataset': ['ndjson'],
    'max_per_category': [
        '70000 --categories pig --lr 0.0005',
        '70000 --categories pig --lr 0.0001',
        # '2000 --categories all --lr 0.0005',
    ],
    'enc_dim': [512],
    'dec_dim': [2048],
    'use_categories_dec': [True],
    'model_type': [
        'retrieverefine --use_layer_norm true --dropout 0.1 --rec_dropout 0.1'
    ],
    # 'lr': [0.0005, 0.0001],

    'sel_k': [1, 2, 4],
    'fixed_mem': [True]#, False]
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='sketchrnn',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    groupname = args.groupname
    base_cmd = CMD + f' --groupname {groupname}'

    grid = GRID

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    free_gpu_max_mem=0.4,
                    email_groupname=args.email_groupname)
