# strokes_to_instruction_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname bigsweep
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname lstm_layernorm
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname imagesweep_images
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname imagesweep_textaug
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname imagesweep_textaug_cnnse
    PYTHONPATH=. python src/models/sweeps/strokes_to_instruction_sweep.py --groupname imagesweep_textaug_rankimgs4
"""

import argparse
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/strokes_to_instruction.py'

NGPUS_PER_RUN = 1

GRID = {
    # Data (and model)
    'drawing_type': ['image'],
    'images': [
        # 'pre,start_to_annotated',  # 7
        'pre,start_to_annotated,full',  # 14
        # 'annotated,full',  # 13
        # 'pre,start_to_annotated,post',  # 5
        # 'pre,annotated,post',  # 8
        # 'annotated',  # 3
    ],

    'cnn_type': [
        'wideresnet',
        # 'se',
        # 'cbam'
    ],

    'data_aug_on_text': [
        'true'
    ],

    # Ranking images auxiliary loss
    # 'rank_imgs_text': ['true'],
    # 'n_rank_imgs': [
    #     2,
    #     4,
    #     8
    # ],
    # 'rank_sim': [
    #     'dot',
    #     'bilinear',
    # ],
    # 'batch_size': [16],

    # Using memory
    'use_mem': [True],
    'base_mem_size': [
        64,
        128,
        256,
    ],
    'mem_dim': [
        64,
        128,
    ],

    # Model
    'dim': [
        64,
        128,
        256,
        # 512,
    ],
    ''
    'model_type': [
        # 'cnn_lstm',
        'lstm',
        # 'lstm --use_layer_norm true',
        # 'transformer_lstm',
    ],
    # 'n_enc_layers': [
    #     '1 --n_dec_layers 1',
    #     '2 --n_dec_layers 2',
    #     '4 --n_dec_layers 4',
    # ],
    'n_dec_layers': [
        # 1,
        # 2,
        4,
    ],

     # Training
    'lr': [
        0.001,
        0.0005,
        # 0.0001,
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='strokes_to_instruction',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname, free_gpu_max_mem=0.2)
