# utils.py

import json
import pickle

#
# I/O
#

def save(obj, fp):
    if fp.endswith('json'):
        with open(fp, 'w') as f:
            json.dump(obj, f)
    elif fp.endswith('pkl'):
        with open(fp, 'wb') as f:
            pickle.dump(obj, f)

def load(fp):
    if fp.endswith('json'):
        with open(fp, 'r') as f:
            return json.load(f)
    elif fp.endswith('pkl'):
        with open(fp, 'rb') as f:
            return pickle.load(f)
