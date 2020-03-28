# sketch_with_plans_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --instruction_set toplevel
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname stack_segs --instruction_set stack
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname decode
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname leaves
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname stack_leaves
    PYTHONPATH=. python src/models/sweeps/sketch_with_plans_sweep.py --groupname load_I2Zenc_pig
"""


import argparse
import copy
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/sketch_with_plans.py'

NGPUS_PER_RUN = 1

BASE_GRID = {
    # data and model
    'categories': [
        # 'all --max_per_category 2000',
        # 'pig --max_per_category 70000',
        'pig --max_per_category 69000',
    ],
    'cond_instructions': [
        'decinputs',
        # 'match --loss_match triplet --enc_dim 2048',
    ],
    'freeze_enc': [
        # True,
        False,
    ],
    'load_enc_and_catemb': [
        'runs/instruction_to_vaez/Mar27_2020/pig_firstsweep/batch_size_32-dataset_ndjson-enc_dim_512-enc_num_layers_1-loss_triplet-lr_0.0001-max_per_category_65000'
    ],

    # training
    'batch_size': [64],
    'lr': [
        0.0005,
        # 0.0001,
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
    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, grid, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname,
                    gpus=[4],
                    free_gpu_max_mem=0.7)
