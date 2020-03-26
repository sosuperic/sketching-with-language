"""
Usage:
    PYTHONPATH=. python src/models/sweeps/instruction_to_vaez_sweep.py --groupname firstsweep
"""

import argparse
import copy
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/instruction_to_vaez.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    'dataset': ['ndjson'],
    'enc_num_layers': [
        '1 --enc_dim 1024',
        '4 --enc_dim 512',
        '4 --enc_dim 256',
    ],
    'lr': [0.0005, 0.0001],
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
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname,
                    free_gpu_max_mem=0.3)
