# experiments.py

"""
Functions to run, manage, analyze experiments

Usage:
    python src/models/core/experiments.py -d runs/strokes_to_instruction/bigsweep
"""

import argparse
from datetime import datetime
import json
import os
import pickle
from pprint import pprint
import schedule
import shutil
import subprocess
import sys
import time
import uuid

import GPUtil
import torch
import torch.nn.functional as F


###################################################################
#
# Experiments, hyperparams and saving experiments data
#
###################################################################

def get_available_GPUs():
    """
    Return list of ints (gpu ids)
    """
    return GPUtil.getAvailable(order='memory', limit=16, maxLoad=0.33, maxMemory=0.33, includeNan=False)

def run_param_sweep(base_cmd, grid, ngpus_per_run=1,
                    prequeue_sleep_secs=120, check_queue_every_nmin=10):
    """
    Launch and queue multiple runs.

    Args:
        base_cmd (str): bash command used to run
        grid (dict): keys are hyperparameters, values are list of grid values
        ngpus_per_run (int): number of gpus to use for each command (> 1 if using multiGPU)
    """
    if ngpus_per_run > 1:
        raise NotImplementedError('MultiGPU per run not handled yet')

    # Generate all commands by creating sets of hyperparameters
    combos = [[]]
    for key, values in grid.items():
        new_combos = []
        for combo in combos:
            for value in values:
                param = f'--{key}={value}'
                new_combo = combo + [param]
                new_combos.append(new_combo)
        combos = new_combos
    for i, combo in enumerate(combos):
        combos[i] = ' '.join(combo)
    print(f'Number of runs: {len(combos)}')

    # get gpus
    system_gpus = GPUtil.getGPUs()
    system_gpu_ids = [gpu.id for gpu in system_gpus]
    available_gpu_ids = get_available_GPUs()
    n_available = len(available_gpu_ids)

    # Run commands on available GPUs
    queued_combos = []
    for i, combo in enumerate(combos):
        if i < n_available:  # run immediately
            gpu_id = available_gpu_ids[i]
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {base_cmd} {combo}'
            if i == 0:
                print(f'Sample command: {cmd}')
            # print(f'{datetime.now()}: {cmd}')
            subprocess.Popen(cmd, shell=True)
        else:
            queued_combos.append(combo)

    # "Queue" the rest
    # let programs start running and utilize the GPU. Some take a long time to initialize the dataset...
    cur_combo_idx = 0
    while True:
        time.sleep(check_queue_every_nmin * 60)
        available_gpu_ids = get_available_GPUs()

        for i in range(len(available_gpu_ids)):  # run on available gpus
            if cur_combo_idx >= len(queued_combos):  # exit if all combos ran
                return

            combo = queued_combos[cur_combo_idx]
            gpu_id = available_gpu_ids[i]
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {base_cmd} {combo}'
            subprocess.Popen(cmd, shell=True)
            cur_combo_idx += 1


def load_hp(hp_obj, dir):
    """
    Args:
        hp_obj: existing HParams object
        dir: directory with existing hp_obj, saved as 'hp.json'

    Returns:
        hp_object updated
    """
    existing_hp = load_file(os.path.join(dir, "hp.json"))
    for k, v in existing_hp.items():
        setattr(hp_obj, k, v)
    return hp_obj


def save_run_data(path_to_dir, hp):
    """
    1) Save stdout to file
    2) Save files to path_to_dir:
        - code_snapshot/: Snapshot of code (.py files)
        - hp.json: dict of HParams object
        - run_details.txt: command used and start time
    """
    print(f"Saving run data to: {path_to_dir}")
    parent_dir = os.path.dirname(path_to_dir)
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.isdir(path_to_dir):
        print("Data already exists in this directory (presumably from a previous run)")
        inp = input(
            'Enter "y" if you are sure you want to remove all the old contents: '
        )
        if inp in ["y", "yes"]:
            print("Removing old contents")
            shutil.rmtree(path_to_dir)
        else:
            print("Exiting")
            raise SystemExit
    print("Creating directory and saving data")
    os.mkdir(path_to_dir)

    # Save snapshot of code
    snapshot_dir = os.path.join(path_to_dir, "code_snapshot")
    print(f"Saving code snapshot to: {snapshot_dir}")
    if os.path.exists(snapshot_dir):  # shutil doesn't work if dest already exists
        shutil.rmtree(snapshot_dir)
    shutil.copytree("src", snapshot_dir)

    # Save hyperparms
    save_file(vars(hp), os.path.join(path_to_dir, "hp.json"), verbose=True)

    # Save some command used to run, start time
    with open(os.path.join(path_to_dir, "run_details.txt"), "w") as f:
        f.write("Command:\n")
        cmd = " ".join(sys.argv)
        start_time = datetime.now().strftime("%B%d_%H-%M-%S")
        f.write(cmd + "\n")
        f.write(f"Start time: {start_time}")
        print("Command used to start program:\n", cmd)
        print(f"Start time: {start_time}")


def create_argparse_and_update_hp(hp):
    """
    Args:
        hp: instance of HParams object
    Returns:
        (updated) hp
        run_name: str (can be used to create directory and store training results)
        parser: argparse object (can be used to add more arguments)
    """

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # Create argparse with an option for every param in hp
    parser = argparse.ArgumentParser()
    for param, default_value in vars(hp).items():
        param_type = type(default_value)
        param_type = str2bool if param_type == bool else param_type
        parser.add_argument(
            "--{}".format(param), dest=param, default=None, type=param_type
        )
    opt, unknown = parser.parse_known_args()

    # Update hp if any command line arguments passed
    # Also create description of run
    run_name = []
    for param, value in sorted(vars(opt).items()):
        if value is not None:
            setattr(hp, param, value)
            if param == "notes":
                run_name = [value] + run_name
            else:
                run_name.append(f"{param}_{value}")
    run_name = "-".join(run_name)
    run_name = ("default_" + str(uuid.uuid4())[:8]) if (run_name == "") else run_name

    return hp, run_name, parser



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
