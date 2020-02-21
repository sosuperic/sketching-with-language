# config.py

from pathlib import Path

##############################################################
#  QUICKDRAW
##############################################################

# Original data
QUICKDRAW_DATA_PATH = Path('data/quickdraw')
NDJSON_PATH = str(QUICKDRAW_DATA_PATH / 'simplified_ndjson' / '{}.ndjson')
NPZ_DATA_PATH = Path('data/quickdraw/npz/')

# Selected categories
CATEGORIES_ANIMAL_PATH = 'data/quickdraw/categories_animals.txt'
CATEGORIES_FINAL_PATH = 'data/quickdraw/categories_final.txt'

# Params and paths for saving various images of drawings
FONT_PATH = str(QUICKDRAW_DATA_PATH / 'ARIALBD.TTF')  # used to label drawing

QUICKDRAW_DRAWINGS_PATH = QUICKDRAW_DATA_PATH / 'drawings'
QUICKDRAW_PAIRS_PATH = QUICKDRAW_DATA_PATH / 'drawings_pairs'
QUICKDRAW_PROGRESSIONS_PATH = QUICKDRAW_DATA_PATH / 'progressions'
QUICKDRAW_PROGRESSIONS_PAIRS_PATH = QUICKDRAW_DATA_PATH / 'progression_pairs_fullinput'
QUICKDRAW_PROGRESSIONS_PAIRS_DATA_PATH = QUICKDRAW_PROGRESSIONS_PAIRS_PATH / 'data'


# For MTurk
S3_PROGRESSIONS_URL = 'https://hierarchical-learning.s3.us-east-2.amazonaws.com/quickdraw/progressions_fullinput/{}/progress/{}'
S3_PROGRESSIONS_PATH = 's3://hierarchical-learning/quickdraw/progressions_fullinput/{}/progress/{}'
S3_PROGRESSION_PAIRS_URL = 'https://hierarchical-learning.s3.us-east-2.amazonaws.com/quickdraw/progression_pairs_fullinput/{}/progress/{}'
S3_PROGRESSION_PAIRS_PATH = 's3://hierarchical-learning/quickdraw/progression_pairs_fullinput/{}/progress/{}'

# MTurk annotated data
ANNOTATED_PROGRESSION_PAIRS_CSV_PATH = QUICKDRAW_PROGRESSIONS_PAIRS_PATH / 'mturk_progressions_pairs_fullresults0.csv'
LABELED_PROGRESSION_PAIRS_PATH = QUICKDRAW_PROGRESSIONS_PAIRS_PATH / 'labeled_progression_pairs'
LABELED_PROGRESSION_PAIRS_DATA_PATH = QUICKDRAW_PROGRESSIONS_PAIRS_PATH / 'labeled_progression_pairs' / 'data'

# Drawings split into pre, current, and post sections.
PRECURRENTPOST_PATH = QUICKDRAW_DATA_PATH / 'precurrentpost'
PRECURRENTPOST_DATA_PATH = PRECURRENTPOST_PATH / 'data'
PRECURRENTPOST_DATAWITHANNOTATIONS_PATH = PRECURRENTPOST_PATH / 'data_with_annotations'
PRECURRENTPOST_DATAWITHANNOTATIONS_SPLITS_PATH = PRECURRENTPOST_PATH / 'data_with_annotations_splits'

# Segmentations (Instruction trees)
SEGMENTATIONS_PATH = QUICKDRAW_DATA_PATH / 'segmentations'

# Annotated
INSTRUCTIONS_VOCAB_DISTRIBUTION_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'vocab_distribution.json'


# Pre-processed splits and data for annotated progerssion pairs
LABELED_PROGRESSION_PAIRS_TRAIN_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'train.pkl'
LABELED_PROGRESSION_PAIRS_VALID_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'valid.pkl'
LABELED_PROGRESSION_PAIRS_TEST_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'test.pkl'

LABELED_PROGRESSION_PAIRS_IDX2TOKEN_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'idx2token.pkl'
LABELED_PROGRESSION_PAIRS_TOKEN2IDX_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'token2idx.pkl'
LABELED_PROGRESSION_PAIRS_IDX2CAT_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'idx2cat.pkl'
LABELED_PROGRESSION_PAIRS_CAT2IDX_PATH = LABELED_PROGRESSION_PAIRS_PATH / 'cat2idx.pkl'


##############################################################
# (Current) best models, experiments, etc.
##############################################################
RUNS_PATH = Path('runs/')

# Models
BEST_STROKES_TO_INSTRUCTION_PATH = 'runs/strokes_to_instruction/Dec18_2019/bigsweep/condition_on_hc_True-dim_256-dropout_0.2-lr_0.0005-model_type_lstm-n_dec_layers_4-n_enc_layers_4-use_categories_dec_True-use_categories_enc_False-use_prestrokes_False'
# BEST_STROKES_TO_INSTRUCTION_PATH = 'best_models/strokes_to_instruction/catsdecoder-dim_512-model_type_cnn_lstm-use_prestrokes_False/'  # OLDER run
BEST_INSTRUCTION_TO_STROKES_PATH = 'runs/instruction_to_strokes/Dec17_2019/cond_instructions_initdec-dec_dim_512-enc_dim_512-lr_0.001-model_type_decodergmm/'

# Segmentations
# BEST_SEG_NDJSON_PATH = SEGMENTATIONS_PATH / 'greedy_parsing/ndjson/nov30_2019/strokes_to_instruction'
BEST_SEG_NDJSON_PATH = SEGMENTATIONS_PATH / 'greedy_parsing/ndjson/Feb20_2020/strokes_to_instruction/strokebasedS2I'
BEST_SEG_PROGRESSION_PAIRS_PATH = SEGMENTATIONS_PATH / 'greedy_parsing/progressionpair/nov26_2019/strokes_to_instruction'