# application.py

"""
Simple app to view outputs.

Directory structure:
    ./static/
    ./static/segmentations  (data)
    ./templates/

Currently implemented:
    - /instruction_tree

Usage:
    PYTHONPATH=. python app/application.py
"""

import os
import random

from flask import Flask, render_template

from config import BEST_SEG_PROGRESSION_PAIRS_PATH, BEST_SEG_NDJSON_PATH

application = Flask(__name__,
                    static_folder='static',
                    template_folder='templates')

def load_seg_trees(seg_dir):
    fns = sorted(os.listdir(seg_dir))
    fps = [os.path.join(seg_dir, fn) for fn in fns]
    segs = []
    for i in range(0, len(fps), 3):
        seg = (fps[i], fps[i+2])  # .jpg, .json, _treant.js
        segs.append(seg)
    return segs

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
    seg_dir = BEST_SEG_PROGRESSION_PAIRS_PATH / 'test'
    print(seg_dir)
    segs = load_seg_trees(seg_dir)
    application.run(host='0.0.0.0', port=8080, debug=False)
