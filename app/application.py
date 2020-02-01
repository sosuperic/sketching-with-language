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
    Deployment: gunicorn -w=2 -b=0.0.0.0:8080 --chdir=app wsgi:application --preload
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
    fns = sorted(os.listdir(STATIC_FOLDER / seg_dir))  # had to add STATIC_FOLDER for gunicorn...
    fps = [os.path.join(seg_dir, fn) for fn in fns]
    segs = []
    for i in range(0, len(fps), 3):
        seg = (fps[i], fps[i+2])  # .jpg, .json, _treant.js
        segs.append(seg)
    return segs

seg_dir = BEST_SEG_PROGRESSION_PAIRS_PATH / 'test'
segs = load_seg_trees(seg_dir)   # by doing gunicorn --preload, loads before forking

#
# Routes
#
@application.route('/instruction_tree', methods=['GET'])
def instruction_tree():
    seg = random.choice(segs)
    img_fp, js_fp = seg
    category = os.path.basename(img_fp).split('_')[0]

    return render_template('index.html',
                            category=category.title(),
                            img_fp=img_fp,
                            js_fp=js_fp)

@application.route('/instruction_tree/<category>', methods=['GET'])
def instruction_tree_category(category):
    cat_segs = [seg for seg in segs if (category in seg[0]) ]
    seg = random.choice(cat_segs)
    img_fp, js_fp = seg

    return render_template('index.html', category=category.title(),
                            img_fp=img_fp, js_fp=js_fp)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8080, debug=False)
