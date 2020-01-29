# strokes_to_instruction_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname bigsweep
"""

import argparse
from src.utils import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/strokes_to_instruction.py'

NGPUS_PER_RUN = 1

GRID = {
    # Model
    'model_type': [
        'cnn_lstm',
        'lstm',
        'transformer_lstm',
    ],
    'dim': [
        256,
        512,
    ],
    'n_enc_layers': [
        '1 --n_dec_layers 1',
        '2 --n_dec_layers 2',
        '4 --n_dec_layers 4',
    ],
    'condition_on_hc': [
        'true',
        'false',
    ],
    'use_prestrokes': ['false'],
    'use_categories_enc': [
        'true',
        'false',
    ],
    'use_categories_dec': [
        'true',
        'false',
    ],
    'dropout': [0.2],
     # Training
    'lr': [
        0.0005,
        0.0001,
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='sketchrnn',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname)å
