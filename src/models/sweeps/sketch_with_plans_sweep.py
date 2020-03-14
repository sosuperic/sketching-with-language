# sketch_with_plans_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --instruction_set toplevel
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname stack_segs --instruction_set stack
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname decode
"""


import argparse
import copy
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_with_plans.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    'dataset': ['ndjson'],
    'prob_threshold': [0],
    'dec_dim': [2048],
    'categories_dim': [256],
    'lr': [
        0.0005,
        0.0001  # should just keep this constant (and same as SketchRNN)
    ],

    'loss_match': [
        'triplet --enc_dim 512',
        # 'decode',
    ],
    'max_per_category': [
        2000,
        # 20000,
    ],
    'instruction_set':  [
        'toplevel --batch_size 16',
        'toplevel_leaves --batch_size 16',
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction_set', default=None)
    parser.add_argument('--groupname', default=None)
    parser.add_argument('--email_groupname', default='sketchwplans',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    grid = BASE_GRID

    groupname = args.groupname
    if groupname is None:
        # Assumes one dataset and one max_per_category at a time right now
        groupname = f"{grid['dataset'][0]}_{args.instruction_set}_{grid['max_per_category'][0]}per"
    base_cmd = CMD + f' --groupname {groupname}'

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname,
                    free_gpu_max_mem=0.3)
