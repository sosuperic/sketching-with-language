"""
Usage:
    PYTHONPATH=. python src/models/sweeps/vaez_to_instruction_sweep.py --groupname pig_firstpass
"""

import argparse
import copy
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/vaez_to_instruction.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    'categories': [
        'pig --max_per_category 65000'
    ],
    'lr': [
        0.0005,
        # 0.0001,
    ],
    'enc_num_layers': [
        '0',                # use z directly
        '1 --enc_dim 256',  # one fc
        '2 --enc_dim 256',  # feedforward w one relu
    ],
    'dec_num_layers': [
        4
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', default=None)
    parser.add_argument('--email_groupname', default='vaez_to_instruction',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    grid = BASE_GRID

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=0, check_queue_every_nmin=0,
                    email_groupname=args.email_groupname,
                    gpus=[5],
                    free_gpu_max_mem=0.8)
