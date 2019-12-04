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
    python application.py
"""

import os
import random

from flask import Flask, render_template

application = Flask(__name__,
                    static_folder='static',
                    template_folder='templates')

SEG_DIR = 'static/segmentations/greedy_parsing/progressionpair/test/'

def load_seg_trees():
    fns = sorted(os.listdir(SEG_DIR))
    fps = [os.path.join(SEG_DIR, fn) for fn in fns]
    segs = []
    for i in range(0, len(fps), 3):
        seg = (fps[i], fps[i+2])  # .jpg, .json, _treant.js
        segs.append(seg)
    return segs

@application.route('/instruction_tree', methods=['GET'])
def instruction_tree():
    seg = random.choice(segs)
    img_fp = seg[0]
    js_fp = seg[1]

    return render_template('index.html',
                            img_fp=img_fp, js_fp=js_fp)

@application.route('/instruction_tree_<category>', methods=['GET'])
# @application.route('/instruction_tree/<category>', methods=['GET'])
def instruction_tree_category(category):
    cur_segs = [seg for seg in segs if (category in seg[0]) ]

    seg = random.choice(cur_segs)
    img_fp = seg[0]
    js_fp = seg[1]

    return render_template('index.html',
                            img_fp=img_fp, js_fp=js_fp)

if __name__ == '__main__':
    segs = load_seg_trees()
    application.run(host='0.0.0.0', port=8080, debug=True)
