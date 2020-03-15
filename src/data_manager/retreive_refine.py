"""


Current problem: Selecting toplevel instruction with highest s2i may not
be great. For example, instructions for lobster don't even include claw...

Usage:
    PYTHONPATH=. python src/data_manager/retreive_refine.py
"""

from collections import defaultdict
import heapq
import numpy as np
import os
from pprint import pprint as pp

from config import RETRIEVAL_SET_PATH, BEST_SEG_NDJSON_PATH
from src.data_manager.quickdraw import ndjson_drawings, save_multiple_strokes_as_img, ndjson_to_stroke3
import src.utils as utils

def create_retrieval_set(N=200, instruction='toplevel_s2iprob'):
    """
    Create a retrieval set by selecting N drawings per category.
    Uses generated instruction trees.

    Args:
        N (int): size of retrieval set per category
        instruction (str): method for extracting instruction
    """

    # Walk over instruction trees
    for root, dirs, fns in os.walk(BEST_SEG_NDJSON_PATH):
        pqueue = []
        category = os.path.basename(root)

        # n = 0
        for fn in fns:
            if (fn != 'hp.json') and fn.endswith('json') and ('treant' not in fn):
                fp = os.path.join(root, fn)
                seg_tree = utils.load_file(fp)
                drawing_id = fn.replace('.json', '')

                if instruction == 'toplevel_s2iprob':
                    text = seg_tree[0]['text']

                heapq.heappush(
                    # cat_to_pqueue[category],
                    pqueue,
                    (seg_tree[0]['score'], drawing_id, text)
                )
                # n += 1
                # if n == 250:
                #     break

        # We are in a directory with seg_trees
        if len(pqueue) > 0:
            print(category)
            # get best instructions
            best = heapq.nlargest(N, pqueue)

            # load drawings
            cat_drawings = ndjson_drawings(category)
            id_to_idx = {d['key_id']: idx for idx, d in enumerate(cat_drawings)}

            # save best
            best_out = []
            for score, id, text in best:
                stroke3 = ndjson_to_stroke3(cat_drawings[id_to_idx[id]]['drawing'])
                out = {
                    'score': score,
                    'id': id,
                    'text': text,
                    'stroke3': stroke3
                }
                best_out.append(out)
            out_fp = RETRIEVAL_SET_PATH / instruction / 'data' / f'{category}.pkl'
            utils.save_file(best_out, out_fp)

            # save a version with just the non-stroke data for easy viewing
            best_out_no_drawing = []
            for d in best_out:
                best_out_no_drawing.append({'score': float(d['score']), 'id': d['id'], 'text': d['text']})
            out_fp = RETRIEVAL_SET_PATH / instruction / 'data' / f'{category}_nodrawing.json'
            utils.save_file(best_out_no_drawing, out_fp)

            # Save drawings
            chunk_n = 25
            for i in range(0, N, chunk_n):
                best_chunk = best_out[i:i+chunk_n]
                drawings = []
                for b in best_chunk:
                    # stroke3 format is in x y deltas, save_multiple_strokes...() expects the actual x y points
                    b['stroke3'][:,0] = np.cumsum(b['stroke3'][:,0])
                    b['stroke3'][:,1] = np.cumsum(b['stroke3'][:,1])
                    drawings.append(b['stroke3'])
                out_dir = RETRIEVAL_SET_PATH / instruction / 'drawings'
                os.makedirs(out_dir, exist_ok=True)
                out_fp = out_dir / f'{category}_{i}-{i+chunk_n}.jpg'
                save_multiple_strokes_as_img(drawings, out_fp)

if __name__ == '__main__':
    create_retrieval_set()