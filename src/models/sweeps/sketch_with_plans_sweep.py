# sketch_with_plans_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --instruction_set toplevel
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname stack_segs --instruction_set stack
"""


import argparse
import copy
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_with_plans.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    'dataset': [
        'ndjson'
    ],
    'max_per_category': [
        # 250,
        2500,
        # 25000,
    ],
    'cond_instructions': [
        # 'initdec',
        # 'decinputs',
        'match',
    ],
    'batch_size': [16],
    'lr': [
        # 0.001,
        0.0005,
        0.0001,
    ],
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
    parser.add_argument('--instruction_set', default=None)
    parser.add_argument('--groupname', default=None)
    parser.add_argument('--email_groupname', default='sketchwplans',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    grid = BASE_GRID
    grid['instruction_set'] = [args.instruction_set]

    groupname = args.groupname
    if groupname is None:
        # Assumes one dataset and one max_per_category at a time right now
        groupname = f"{grid['dataset'][0]}_{args.instruction_set}_{grid['max_per_category'][0]}per"
    base_cmd = CMD + f' --groupname {groupname}'

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname,
                    free_gpu_max_mem=0.3)
