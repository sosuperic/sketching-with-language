# experiments.py

"""
Functions to run, manage, analyze experiments

Usage:
    python src/models/core/experiments.py -mvd runs/sketchrnn
"""

import argparse
import os


#################################################
#
# Scripts
#
#################################################

def find_min_valid(dir):
    """
    Assumes each run has file with the name: 'e<epoch>_loss<loss>.pt'

    Args:
        dir (str)

    TODO: this could be made more general purpose by searching within logs
    (stdout.txt); can search for minimum or maximum of a certain metric
    """
    min_valid = float('inf')
    min_run = None
    for root, dirs, fns in os.walk(dir):
        for fn in fns:
            if fn.endswith('pt') and ('loss' in fn):
                loss = float(fn.split('loss')[1].strip('.pt'))
                if loss < min_valid:
                    min_valid = loss
                    min_run = root
    print(f'{min_valid}: {min_run}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mvd', '--min_valid_dir', default=None, help='find best run within this directory')
    args = parser.parse_args()

    if args.min_valid_dir:
        find_min_valid(args.min_valid_dir)
