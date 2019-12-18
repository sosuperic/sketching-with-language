# sketch_rnn_sweep.py

"""
Usage:
    PYTHONPATH=. python src/models/sweeps/instruction_to_strokes_sweep.py --groupname testsweep
"""

import argparse
from src.utils import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/instruction_to_strokes.py'

NGPUS_PER_RUN = 1

GRID = {
    # InstructionToStrokes
    'cond_instructions': [
        'initdec --dec_dim 512',  # must be equal to enc_dim
        'decinputs --dec_dim 256',
        'decinputs --dec_dim 512',
        'decinputs --dec_dim 1024',
    ],
    # SketchRNN
    'model_type': ['decodergmm'],
    'enc_dim': [
        # TODO: since we're loading the text embeddings from a StrokeToInstruction model,
        # we must use the same dimension. (There is no projection from word embeddings to
        # input_dim in the InstructionEncoder model).
        512
    ],
     # Training
    'lr': [
        0.001,
        0.0005,
        0.0001,
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    args = parser.parse_args()

    base_cmd = CMD + f' --groupname {args.groupname}'

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_secs=10, check_queue_every_nmin=10)
