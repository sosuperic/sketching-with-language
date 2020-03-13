"""
Usage:
PYTHONPATH=. python src/models/sweeps/contrastive_drawing_encoder_sweep.py --groupname firstsweep
"""

import argparse
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/contrastive_drawing_encoder.py'

NGPUS_PER_RUN = 1

GRID = {
    'model_type': ['lstm'],
    'n_enc_layers': [4],


    'loss_tau': [
        0.1,
        1.0,
    ],
    'max_per_category': [
        '7000 --lr 0.0005',
        '7000 --lr 0.0001',
        '70000 --lr 0.0001',
    ],
    'dim': [
        128,
        256,
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='instruction_to_strokes',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    free_gpu_max_mem=0.67,
                    email_groupname=args.email_groupname)
