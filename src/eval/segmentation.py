# segmentation.py

"""
Usage:
    PYTHONPATH=. python src/eval/segmentation.py

Treant: https://fperucic.github.io/treant-js/
"""
import os

from src.data_manager.quickdraw import SEGMENTATIONS_PATH
import src.utils as utils

def convert_all_segmentations_to_treants(seg_dir):
    for root, dirs, fns in os.walk(seg_dir):
        for fn in fns:
            if fn.endswith('json') and ('treant' not in fn):
                fp = os.path.join(root, fn)
                seg_tree = utils.load_file(fp)
                out_fp = fp.replace('.json', '_treant.js')
                save_segmentation_in_treant_format(seg_tree, out_fp)

def save_segmentation_in_treant_format(seg_tree, out_fp):
    """[summary]

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


if __name__ == "__main__":
    # seg_dir = SEGMENTATIONS_PATH / 'greedy_parsing' / 'progressionpair'
    seg_dir = SEGMENTATIONS_PATH / 'greedy_parsing' / 'progressionpair/instruction_to_strokes/dec19/test'
    convert_all_segmentations_to_treants(seg_dir)
