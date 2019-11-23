# utils.py

import argparse
from datetime import datetime
import json
from nltk.tokenize import (
    word_tokenize,
)
import os
import pickle
import shutil
import sys
import torch
import torch.nn.functional as F
import uuid


#################################################
#
# Simple I/O
#
#################################################


def save_file(data, path, verbose=False):
    """Creates intermediate directories if they don't exist."""
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if verbose:
        print(f"Saving: {path}")

    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=2)
    elif ext == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=4, separators=(",", ": "), sort_keys=True)
            f.write("\n")  # add trailing newline for POSIX compatibility


def load_file(path):
    _, ext = os.path.splitext(path)
    if ext == ".pkl":
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    return data


#################################################
#
# Data
#
#################################################


def normalize_sentence(sentence):
    """
    Args:
        sentence: str
    """
    return word_tokenize(sentence.lower())


#################################################
#
# Hyperparams and saving experiments data
#
#################################################


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
        if inp == "y":
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
