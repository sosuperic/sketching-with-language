# application.py

"""
Simple app to view outputs.

Directory structure:
    ./static/
    ./static/data -> ../data  (symlink)
    ./templates/

Currently implemented:
    - /instruction_tree

Usage:
    Deployment: gunicorn -w=2 -b=0.0.0.0:8080 --chdir=app wsgi:application --preload <--daemon>
    Development: PYTHONPATH=. python app/application.py
        - currently broken
"""

import os
import random

from flask import Flask, render_template

from config import BEST_SEG_PROGRESSION_PAIRS_PATH, BEST_SEG_NDJSON_PATH

STATIC_FOLDER = 'static'
TEMPLATE_FOLDER = 'templates'

application = Flask(__name__,
                    static_folder=STATIC_FOLDER,
                    template_folder=TEMPLATE_FOLDER)
application.url_map.strict_slashes = False

def load_seg_trees(seg_dir):
    """
    Args:
        seg_dir (str): [description]

    Returns:
        dict: keys are (category, id), values are (img_fp, treant_js_fp)
    """
    fns = sorted(os.listdir(os.path.join(STATIC_FOLDER, seg_dir)))  # had to add STATIC_FOLDER for gunicorn...
    fps = [os.path.join(seg_dir, fn) for fn in fns]

    segs = {}
    for i in range(0, len(fps), 3):
        img_fp, seg_fp, treant_fp = fps[i], fps[i+1], fps[i+2]  # .jpg, .json, _treant.js
        img_fn = os.path.basename(img_fp)
        category, id = img_fn.split('.')[0].split('_')
        segs[(category, id)] = (img_fp, treant_fp)
    return segs

# by doing gunicorn --preload, loads before forking
# seg_dir1 = BEST_SEG_PROGRESSION_PAIRS_PATH / 'train'
seg_dir1 = 'data/quickdraw/segmentations/greedy_parsing/progressionpair/Feb05_2020/strokes_to_instruction/rerun_copy_threshold0.55/train'
seg_dir2 = 'data/quickdraw/segmentations/greedy_parsing/progressionpair/Feb05_2020/strokes_to_instruction/rerun/train'
segs1 = load_seg_trees(seg_dir1)
segs2 = load_seg_trees(seg_dir2)

#
# Routes
#
@application.route('/instruction_tree', methods=['GET'])
def instruction_tree():
    key = random.choice(list(segs1.keys()))
    category, id = key

    img_fp, js_fp1 = segs1[key]
    _, js_fp2 = segs2[key]

    return render_template('index.html',
                            category=category.title(),
                            img_fp=img_fp, js_fp1=js_fp1, js_fp2=js_fp2)

@application.route('/instruction_tree/<category>', methods=['GET'])
def instruction_tree_category(category):
    keys_for_category = [key for key in segs1.keys() if (category == key[0])]
    key = random.choice(keys_for_category)
    category, id = key

    img_fp, js_fp1 = segs1[key]
    _, js_fp2 = segs2[key]

    return render_template('index.html',
                            category=category.title(),
                            img_fp=img_fp, js_fp1=js_fp1, js_fp2=js_fp2)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080, debug=False)
