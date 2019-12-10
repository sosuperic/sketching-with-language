# sketch_rnn_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_rnn_sweep.py --groupname testsweep
"""

import argparse
from src.utils import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_rnn.py'

NGPUS_PER_RUN = 1

GRID = {
    'model_type': ['decodergmm'],
    'dataset': ['progressionpair'],
    'lr': [0.0005, 0.0001],
    # 'enc_dim': ['256 --dec_dim 512', '512 --dec_dim 2048'],
    'enc_dim': [
        '256 --dec_dim 256',
        '512 --dec_dim 512',
        '1024 --dec_dim 1024',
        '2048 --dec_dim 2048'
        ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_secs=10, check_queue_every_nmin=10)
