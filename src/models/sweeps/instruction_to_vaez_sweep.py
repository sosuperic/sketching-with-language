"""
Usage:
    PYTHONPATH=. python src/models/sweeps/instruction_to_vaez_sweep.py --groupname pig_firstsweep
"""

import argparse
import copy
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/instruction_to_vaez.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    'dataset': ['ndjson --max_per_category 65000'],
    'batch_size': [32],
    'enc_num_layers': [
        '4 --enc_dim 256',
        # '1 --enc_dim 512',
    ],
    'loss': ['triplet'],
    'lr': [
        # 0.0001,
        0.00005,
    ],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', default=None)
    parser.add_argument('--email_groupname', default='instruction_to_vaez',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    grid = BASE_GRID

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=0, check_queue_every_nmin=0,
                    email_groupname=args.email_groupname,
                    gpus=[6],
                    free_gpu_max_mem=0.85)
