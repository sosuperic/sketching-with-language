# segmentation_sweep.py


"""
Used to "parallelize" segmenting over larger ndjson dataset.

By using final_categories() as the grid for categories, each job
segments one category.

Usage:
    PYTHONPATH=. python src/models/sweeps/segmentation_sweep.py --groupname parentchildsim
"""

import argparse

from src.data_manager.quickdraw import final_categories
from src.models.core.experiments import run_param_sweep

CMD = 'PYTHONPATH=. python src/models/segmentation.py --segment_dataset ndjson'

NGPUS_PER_RUN = 1



GRID = {
    # Ndjson data
    'categories': final_categories(),
    'max_per_category': [
        2750,
    ],
    # Segmenation model
    'split_scorer': ['strokes_to_instruction'],
    'score_parent_child_text_sim': [
        'true',
    ],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupname', help='name of subdir to save runs')
    parser.add_argument('--email_groupname', default='segmentation',
                        help='Sent in email when sweep is completed.')
    args = parser.parse_args()

    groupname = args.groupname
    base_cmd = CMD + f' --groupname {groupname}'
    # print(base_cmd)

    run_param_sweep(base_cmd, GRID, ngpus_per_run=NGPUS_PER_RUN,
                    prequeue_sleep_nmin=10, check_queue_every_nmin=10,
                    email_groupname=args.email_groupname)
