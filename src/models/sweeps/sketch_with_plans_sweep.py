# sketch_with_plans_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --instruction_set toplevel
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --instruction_set toplevel_leaves
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --instruction_set stack
"""


import argparse
import copy
from src.utils import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_with_plans.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    'cond_instructions': ['initdec', 'decinputs'],
    'lr': [
        0.0005,
        0.0001,
    ],
    'enc_dim': [
        '256 --dec_dim 256',
        '512 --dec_dim 512',
        # '1024 --dec_dim 1024',
        # '2048 --dec_dim 2048'
    ],
}

GRID_1 = copy.deepcopy(BASE_GRID)
GRID_1.update({
    'instruction_set': ['toplevel'],
})

GRID_2 = copy.deepcopy(BASE_GRID)
GRID_2.update({
    'instruction_set': ['toplevel_leaves'],
})

GRID_3 = copy.deepcopy(BASE_GRID)
GRID_3.update({
    'instruction_set': ['stack'],
    'cond_instructions': ['decinputs'],
    'lr': [
        0.001,
        0.0005,
        0.0001,
    ],
    'enc_dim': [
        '32 --batch_size 32 --dec_dim 256',
        '64 --batch_size 16 --dec_dim 256',
        '64 --batch_size 16 --dec_dim 512',
        '128 --batch_size 16 --dec_dim 256',
    ],
})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction_set')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.instruction_set}'

    if args.instruction_set == 'toplevel':
        grid = GRID_1
    elif args.instruction_set == 'toplevel_leaves':
        grid = GRID_2
    elif args.instruction_set == 'stack':
        grid = GRID_3

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_secs=10, check_queue_every_nmin=10)
