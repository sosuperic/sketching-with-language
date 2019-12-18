# experiments.py

"""
Functions to run, manage, analyze experiments

Usage:
    python src/models/core/experiments.py -d runs/strokes_to_instruction/bigsweep
"""

import argparse
import os


#################################################
#
# Scripts
#
#################################################

def find_min_valid(dir, best_n=20):
    """
    Assumes each run has file with the name: 'e<epoch>_loss<loss>.pt'.

    Args:
        dir (str)
        best_n (int): print best_n runs in sorted order

    TODO: this could be made more general purpose by searching within logs
    (stdout.txt); can search for minimum or maximum of a certain metric
    """
    print(f'Looking in: {dir}\n')

    # Get losses
    min_valid = float('inf')
    runs_losses = []
    for root, dirs, fns in os.walk(dir):
        for fn in fns:
            # Get loss from model fn (e<epoch>_loss<loss>.pt)
            if fn.endswith('pt') and ('loss' in fn):
                loss = float(fn.split('loss')[1].strip('.pt'))
                run = root.replace(dir + '/', '')
                runs_losses.append((run, loss))

    # Print
    for run, loss in sorted(runs_losses, key=lambda x: x[1])[:best_n][::-1]:
        print(f'{loss}: {run}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default=None, help='find best run within this directory')
    args = parser.parse_args()

    find_min_valid(args.dir)
