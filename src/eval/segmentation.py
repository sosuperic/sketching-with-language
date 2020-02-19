"""
1. Converts the instruction tree files generated by src/models/segmentation.py
to a format suitable for visualization with the treant-js library. These
files are currently used by the app/application.py.
(Treant: https://fperucic.github.io/treant-js/)

2. Score instruction trees

Usage:
    PYTHONPATH=. python src/eval/segmentation.py --treant -d <dir>
    PYTHONPATH=. python src/eval/segmentation.py --score_split -d <dir>
    PYTHONPATH=. python src/eval/segmentation.py --score_match -d <dir>
"""

import argparse
from collections import defaultdict
import os
from pprint import pprint

import numpy as np
import pandas as pd

from config import SEGMENTATIONS_PATH, ANNOTATED_PROGRESSION_PAIRS_CSV_PATH
from src.eval.strokes_to_instruction import InstructionScorer
from src.models.segmentation import prune_seg_tree
import src.utils as utils

###############################################################
#
# Generating trees in treant.js format for webapp
#
###############################################################

def save_segmentation_in_treant_format(seg_tree, out_fp):
    """
    Convert instructino tree generated by src/models/segmentation.py into
    a js file that can be loaded by treant.js

    Args:
        seg ([list of dicts]): represents hierarchical segmentation and instructions for a sketch
    """
    PARENT_NODE_FMT = """
    var node_{} = {{
        text: {{ name: "{}-{}: {}" }}
    }}
    """

    NODE_FMT = """
    var node_{} = {{
        parent: {},
        text: {{ name: "{}-{}: {}" }}
    }}
    """

    CONFIG_FMT = """
    var simple_chart_config = [
        config, {}
    ];
    """
    with open(out_fp, 'w') as f:
        node_names = []

        # write parent node
        seg = seg_tree[0]
        name = seg['id']
        node_names.append('node_' + name)
        parent = PARENT_NODE_FMT.format(name, seg['left'], seg['right'], seg['text'])
        f.write(parent + '\n')

        # Write all the child nodes
        for i in range(1, len(seg_tree)):
            seg = seg_tree[i]
            name = seg['id']
            node_names.append('node_' + name)
            par_name = 'node_' + seg['parent']
            node = NODE_FMT.format(name, par_name, seg['left'], seg['right'], seg['text'])
            f.write(node + '\n')

        # Write the simple_chart_config
        f.write(CONFIG_FMT.format(',\n'.join(node_names)))

def convert_all_segmentations_to_treants(seg_dir, prob_threshold):
    """
    Recursively walk through directory and find instruction trees (i.e. segmentations)
    generated by src/model/segmentation.py. For each one,

    Args:
        seg_dir (str):
    """
    for root, dirs, fns in os.walk(seg_dir):
        for fn in fns:
            if (fn != 'hp.json') and fn.endswith('json') and ('treant' not in fn):
                fp = os.path.join(root, fn)
                seg_tree = utils.load_file(fp)
                # TODO: save prob_threshold in filename?
                out_fp = fp.replace('.json', '_treant.js')
                pruned_seg_tree = prune_seg_tree(seg_tree, prob_threshold)
                n_og, n_pruned = len(seg_tree), len(pruned_seg_tree)
                # pprint(seg_tree)
                # pprint(pruned_seg_tree)
                print(f'N segments before vs. after pruning: {n_og}, {n_pruned}')
                save_segmentation_in_treant_format(pruned_seg_tree, out_fp)


###############################################################
#
# Score Instruction Trees
#
###############################################################

def score_segtree_on_parent_child_splits(seg_dir):

    def map_parents_to_children(seg_tree):
        """seg_tree is list of dicts"""
        id_to_node = {}
        parid_to_childids = defaultdict(list)
        for node in seg_tree:
            id, parid = node['id'],  node['parent']
            id_to_node[id] = node
            if parid != '':  # root node
                parid_to_childids[parid].append(id)
        return id_to_node, parid_to_childids

    def calc_seg_score(id_to_node, parid_to_childids, scorers):
        metric2scores = defaultdict(list)
        for parid, childids in parid_to_childids.items():
            par_text = id_to_node[parid]['text']
            child_text_concat = ' '.join([id_to_node[childid]['text'] for childid in childids])

            for scorer in scorers:
                for metric, value in scorer.score(par_text, child_text_concat).items():
                    metric2scores[metric].append(value)

        metric2scores = {metric: np.mean(scores) for metric, scores in metric2scores.items()}
        return metric2scores


    scorers = [InstructionScorer('bleu'), InstructionScorer('rouge')]

    metric2allscores = defaultdict(list)
    for root, dirs, fns in os.walk(seg_dir):
        for fn in fns:
            if (fn != 'hp.json') and fn.endswith('json') and ('treant' not in fn):
                fp = os.path.join(root, fn)
                seg_tree = utils.load_file(fp)

                # calculate score for this tree
                id_to_node, parid_to_childids = map_parents_to_children(seg_tree)
                metric2scores = calc_seg_score(id_to_node, parid_to_childids, scorers)
                for metric, score in metric2scores.items():
                    metric2allscores[metric].append(score)

    metric2allscores_mean = {metric: np.mean(scores) for metric, scores in metric2allscores.items()}
    metric2allscores_std = {metric: np.std(scores) for metric, scores in metric2allscores.items()}

    print('-' * 100)
    print(f'Scores for: {seg_dir}')
    print('Mean:')
    pprint(metric2allscores_mean)
    print()
    print('Std:')
    pprint(metric2allscores_std)

def score_segtree_match_with_annotations(seg_dir):
    def load_annotations():
        """
        Returns:
            dict: drawing_id -> row from dataframe of Mturk annotations
        """
        df = pd.read_csv(ANNOTATED_PROGRESSION_PAIRS_CSV_PATH)
        id_to_annotations = {}
        for i, row in df.iterrows():
            drawing_id = row['Input.id']
            id_to_annotations[drawing_id] = row
        return id_to_annotations

    def calc_seg_score(drawing_id, seg_tree, id_to_annotations, scorers):
        """
        Calculate score for one tree.

        Args:
            drawing_id (int)
            seg_tree (list of dicts): [description]
            id_to_annotations (dict): drawing_id (int) -> row from dataframe of Mturk annotations
            scorers (list): Scorers (bleu, rouge)
        """
        annotations = id_to_annotations[drawing_id]  # TODO: check that this exists...

        category = annotations['Input.category']
        gt_instruction = annotations['Answer.annotation'].replace('\r', '')
        ndjson_start = annotations['Input.start']
        ndjson_end = annotations['Input.end']
        n_segs = annotations['Input.n_segments']
        url = annotations['Input.url']

        metric2score = {}
        match = None
        # There may be one segment within the instruction tree that matches the annotated segment
        for node in seg_tree:
            if (node['left'] == ndjson_start) and (node['right'] == ndjson_end):  # TODO: check offsets etc.
                gen_instruction = node['text']
                for scorer in scorers:
                    for metric, value in scorer.score(gt_instruction, gen_instruction).items():
                        metric2score[metric] = value

                match = {
                    'id': drawing_id,
                    'gen_instruction': gen_instruction,
                    'gt_instruction': gt_instruction,
                    'category': category,
                    'url': url,

                    # 'metric2score': metric2score,
                }

        return metric2score, match


    scorers = [InstructionScorer('bleu'), InstructionScorer('rouge')]
    metric2allscores = defaultdict(list)
    all_matches = []
    id_to_annotations = load_annotations()
    n_segs = 0

    # Find instruction trees
    for root, dirs, fns in os.walk(seg_dir):
        for fn in fns:
            if (fn != 'hp.json') and fn.endswith('json') and ('treant' not in fn):
                fp = os.path.join(root, fn)
                drawing_id = fn.split('_')[1].replace('.json', '') # fn: lion_6247028344487936.jpg
                drawing_id = int(drawing_id)
                seg_tree = utils.load_file(fp)

                # calculate score for this tree
                metric2score, match = calc_seg_score(drawing_id, seg_tree, id_to_annotations, scorers)
                if match:
                    all_matches.append(match)
                    for metric, score in metric2score.items():
                        metric2allscores[metric].append(score)

                n_segs += 1

    metric2allscores_mean = {metric: np.mean(scores) for metric, scores in metric2allscores.items()}
    metric2allscores_std = {metric: np.std(scores) for metric, scores in metric2allscores.items()}

    print('-' * 100)
    print(f'Number of matches: {len(all_matches)} / {n_segs}')

    print(f'Scores for: {seg_dir}')
    print('Mean:')
    pprint(metric2allscores_mean)
    print()
    print('Std:')
    pprint(metric2allscores_std)

    # TODO: save all_matches?
    # pprint(all_matches[:100])




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--seg_dir', default=None,
                        help='find all segmentation files within this directory')
    parser.add_argument('--treant', action='store_true')
    parser.add_argument('-pt', '--prob_threshold', type=float, default=0.0)

    parser.add_argument('--score_split', action='store_true')
    parser.add_argument('--score_match', action='store_true')
    args = parser.parse_args()

    if args.treant:
        convert_all_segmentations_to_treants(args.seg_dir, args.prob_threshold)
    if args.score_split:
        score_segtree_on_parent_child_splits(args.seg_dir)
    if args.score_match:
        score_segtree_match_with_annotations(args.seg_dir)