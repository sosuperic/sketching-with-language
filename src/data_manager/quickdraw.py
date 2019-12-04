"""
Analyzing and preparing QuickDraw data

Original data obtained from: https://github.com/googlecreativelab/quickdraw-dataset


ndjson format: list (A) of lists (B) of two lists (C)
    - Each list B corresponds to one segment of the drawing until a "penup" point
    - Each list C corresponds to the x or y points in that segment

Stroke-3 format: (delta-x, delta-y, binary for if pen is lifted)
Stroke-5 format: consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
    one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.

"""

import argparse
from collections import defaultdict, Counter
import csv
import json
import math
import os
from pathlib import Path
from pprint import pprint
import random
import subprocess
from time import sleep

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cairocffi as cairo
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFont, ImageDraw

import src.utils as utils

###################################################################
#
# Config and paths
#
###################################################################

# Original
QUICKDRAW_DATA_PATH = Path('data/quickdraw')
NDJSON_PATH = str(QUICKDRAW_DATA_PATH / 'simplified_ndjson' / '{}.ndjson')

# Selected categories
CATEGORIES_ANIMAL_PATH = 'data/quickdraw/categories_animals.txt'
CATEGORIES_FINAL_PATH = 'data/quickdraw/categories_final.txt'

# Params and paths for saving various images of drawings
SIDE = 112
LINE = 6
PAIRS_MIN_STROKES = 3
FONT_PATH = str(QUICKDRAW_DATA_PATH / 'ARIALBD.TTF')

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


###################################################################
#
# Utilities
#
###################################################################

def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0, 0, 0), fg_color=(1, 1, 1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """

    original_side = 256.

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1, 1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)

    return raster_images

def ndjson_drawings(category):
    """Return list of ndjson data"""
    path = NDJSON_PATH.format(category)
    drawings = open(path, 'r').readlines()
    drawings = [json.loads(d) for d in drawings]
    drawings = [d for d in drawings if d['recognized']]
    return drawings

def animal_categories():
    """Return list of strings"""
    categories = open(CATEGORIES_ANIMAL_PATH).readlines()
    categories = [c.strip() for c in categories]
    return categories

def final_categories():
    categories = open(CATEGORIES_FINAL_PATH).readlines()
    categories = [c.strip() for c in categories]
    return categories

def build_category_index(data):
    """
    Returns mappings from index to category and vice versa.

    Args:
        data: list of dicts, each dict is one sample
    """
    categories = set()
    for sample in data:
        categories.add(sample['category'])
    categories = sorted(list(categories))
    idx2cat = {i: cat for i, cat in enumerate(categories)}
    cat2idx = {cat: i for i, cat in idx2cat.items()}

    return idx2cat, cat2idx

def ndjson_to_stroke3(ndjson_format):
    """
    Parse an ndjson sample and return ink (as np array) and classname.

    Taken from https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/recurrent_quickdraw.md

    Args:
        ndjson_format: drawing in ndjson format

    Returns: [len, 3] np array
    """
    stroke_lengths = [len(stroke[0]) for stroke in ndjson_format]
    total_points = sum(stroke_lengths)
    stroke3 = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    for stroke in ndjson_format:
        for i in [0, 1]:
            stroke3[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        stroke3[current_t - 1, 2] = 1  # stroke_end

    # Size normalization
    lower = np.min(stroke3[:, 0:2], axis=0)
    upper = np.max(stroke3[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    stroke3[:, 0:2] = (stroke3[:, 0:2] - lower) / scale

    # Compute deltas
    stroke3[1:, 0:2] -= stroke3[0:-1, 0:2]
    stroke3 = stroke3[1:, :]

    return stroke3

def stroke3_to_stroke5(seq, max_len=None):
    """
    Convert from stroke-3 to stroke-5 format
    "to_big_strokes()" in Magenta's sketch_rnn/utils.py

    Args:
        seq: [len, 3] float array

    Returns:
        result: [max_len, 5] float array
        l: int, length of sequence
    """
    result_len = max_len if max_len else len(seq)
    result = np.zeros((result_len, 5), dtype=float)
    l = len(seq)
    assert l <= result_len
    result[0:l, 0:2] = seq[:, 0:2]  # 1st and 2nd values are same
    result[0:l, 3] = seq[:, 2]  # stroke-5[3] = pen-up, same as stroke-3[2]
    result[0:l, 2] = 1 - result[0:l, 3]  # stroke-5[2] = pen-down, stroke-3[2] = pen-up (so inverse)
    result[l:, 4] = 1  # last "stroke" has stroke5[4] equal to 1, all other values 0 (see Figure 4); hence l
    return result

def normalize_strokes(data, scale_factor=None, scale_factor_key='stroke3',
                      stroke_keys=['stroke3']):
    """
    Normalize stroke3 or stroke5 dataset's delta_x and delta_y values.

    Args:
        data: list of dicts, each dict is one sample that contains stroke information
        scale_factor: float if given
        scale_factor_key: str
        stroke_keys: keys in data's dicts that must be scaled (e.g. stroke3, stroke3_segment)
    """
    if scale_factor is None:
        scale_factor = _calculate_normalizing_scale_factor(data, scale_factor_key)
    print('Scale factor: ', scale_factor)

    normalized_data = []
    for sample in data:
        for key in stroke_keys:
            stroke = sample[key]
            stroke[:, 0:2] /= scale_factor
            sample[key] = stroke
        normalized_data.append(sample)

    return normalized_data

def _calculate_normalizing_scale_factor(data, scale_factor_key):  #
    """
    Calculate the normalizing factor in Appendix of paper
    (calculate_normalizing_scale_factor() in Magenta's sketch_rnn/utils.py)

    Args:
        data: list of dicts
        scale_factor_key: str
    """
    deltas = []
    for sample in data:
        # TODO: should I calculate this scale factor based only on stroke3_**segment**?? Or stroke3
        stroke = sample[scale_factor_key]
        for j in range(stroke.shape[0]):
            deltas.append(stroke[j][0])
            deltas.append(stroke[j][1])
    deltas = np.array(deltas)
    scale_factor = np.std(deltas)
    return scale_factor


def save_strokes_as_img(sequence, output_fp):
    """
    Args:
        sequence: [len, 3] np array
            [x, y, pen up] (this is x-y positions, not delta x and delta y's)
        output_fp: str
    """
    strokes = np.split(sequence, np.where(sequence[:, 2] == 1)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    pil_image.save(output_fp)
    plt.close('all')

def convert_stroke5_to_ndjson_seq(stroke5):
    """
    TODO: this is a WIP.
        - may have to unnormalize stroke5 before
        - What should the offset be? stroke5 contains positive and negative numbers for x / y, ndjson_seq doesn't
        - Possible reference: https://github.com/hardmaru/sketch-rnn-datasets/blob/master/draw_strokes.py

    Args:
        stroke5: [len, 5] numpy array

    Returns: drawing in ndjson format
    """
    ndjson_seq = []
    pen_up = np.where(stroke5[:,3] == 1)[0].tolist()
    pen_up = [0] + pen_up

    cur_x, cur_y = 0, 0
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    # cur_x, cur_y = 123, 123   # TODO: shouldn't have negative values... hacking it right now
    for seg_idx in range(len(pen_up) - 1):
        seg_x = []
        seg_y = []
        start_stroke_idx, end_stroke_idx = pen_up[seg_idx], pen_up[seg_idx+1]
        for stroke_idx in range(start_stroke_idx, end_stroke_idx + 1):

            dx, dy = stroke5[stroke_idx, 0], stroke5[stroke_idx, 1]

            # TODO: this is just some manual super hack right now to get the positions to show up
            dx /= 2
            dy /= 2

            cur_x += dx
            cur_y += dy
            seg_x.append(cur_x)
            seg_y.append(cur_y)

            # min_x = min(cur_x, min_x)
            # min_y = min(cur_x, min_y)
            # max_x = max(cur_x, max_x)
            # max_y = max(cur_x, max_y)
        ndjson_seq.append([seg_x, seg_y])

    return ndjson_seq

def create_progression_image_from_ndjson_seq(strokes):
    """
    Args:
        strokes: list (A) of list (B) of lists (C)
            - A is of length n_penups
            - B is the ith segment
            - B contains two lists (C's) one for each x and y

    Returns: Image
    """

    font_size = int(SIDE * 0.2)
    font_space = 2 * font_size  # space for numbering
    font = ImageFont.truetype(FONT_PATH, font_size)
    segs_in_row = 8
    border = 3


    n_segs = len(strokes)
    x_segs = n_segs if (n_segs < segs_in_row) else segs_in_row
    y_segs = math.ceil(n_segs / segs_in_row)
    x_height = SIDE * x_segs + border * (x_segs + 1)
    y_height = SIDE * y_segs + border * (y_segs + 1) + (font_space) * y_segs
    img = Image.new('L', (x_height, y_height))
    img.paste(255, [0, 0, img.size[0], img.size[1]])  # fill in image with white
    for s_idx, s in enumerate(strokes):
        segments = strokes[:s_idx + 1]
        vec = vector_to_raster([segments], side=SIDE, line_diameter=LINE)[0]
        seg_vec = vec.reshape(SIDE, SIDE)

        seg_img = Image.fromarray(seg_vec, 'L')
        seg_img = ImageOps.expand(seg_img, (0, font_space, 0, 0))  # add white space above for number
        seg_img = ImageOps.invert(seg_img)
        seg_img = ImageOps.expand(seg_img, border=border, fill='gray')
        num_offset = 0.5 * font_size
        draw = ImageDraw.Draw(seg_img)
        draw.text((num_offset, num_offset), str(s_idx + 1), (0), font=font)

        x_offset = (s_idx % segs_in_row) * (border + SIDE)
        y_offset = (s_idx // segs_in_row) * (border + font_space + SIDE)
        img.paste(seg_img, (x_offset, y_offset))

    return img

###################################################################
#
# Saving and manipulating raw data
#
###################################################################

def save_pairs(n=None):
    """
    Save random pairs of drawings
    """
    categories = animal_categories()
    for cat in categories:
        print(cat)
        drawings = ndjson_drawings(cat)
        for d_idx in range(0, len(drawings), 2):
            if (n is not None) and (d_idx == n):
                break

            # get drawing
            d1 = drawings[d_idx]
            d2 = drawings[d_idx + 1]
            strokes1 = d1['drawing']
            strokes2 = d2['drawing']

            # convert strokes to image and combine into one image
            img_vec1 = vector_to_raster([strokes1], side=SIDE, line_diameter=LINE)[0]
            img_vec2 = vector_to_raster([strokes2], side=SIDE, line_diameter=LINE)[0]
            pair = [img_vec1.reshape(SIDE, SIDE), img_vec2.reshape(SIDE, SIDE)]
            pair = np.hstack(pair)
            img = Image.fromarray(pair, 'L')
            img = ImageOps.invert(img)

            # save image
            out_dir = QUICKDRAW_PAIRS_PATH / cat
            os.makedirs(out_dir, exist_ok=True)
            out_fp = out_dir / f'{d_idx}.jpg'
            img.save(out_fp)

def save_final_drawings(n=None):
    """
    Convert stroke data to raster image and save. Used to filter out bad drawings
    """
    categories = animal_categories()
    for cat in categories:
        print(cat)
        drawings = ndjson_drawings(cat)
        for d_idx, d in enumerate(drawings):
            if (n is not None) and (d_idx == n):
                break

            id = d['key_id']
            strokes = d['drawing']

            img_vec = vector_to_raster([strokes], side=SIDE, line_diameter=LINE)[0]
            img = Image.fromarray(img_vec.reshape(SIDE, SIDE), 'L')
            img = ImageOps.invert(img)
            out_dir = QUICKDRAW_DRAWINGS_PATH / cat
            os.makedirs(out_dir, exist_ok=True)
            img.save(out_dir / f'{id}.jpg')

def save_progressions(n=None):
    """
    Saw progressions of strokes of data
    """
    categories = animal_categories()
    for cat in categories:
        print(cat)

        # make directories
        out_dir_base = QUICKDRAW_PROGRESSIONS_PATH
        out_dir_progress = out_dir_base / cat / 'progress'
        out_dir_meta = out_dir_base / cat / 'meta'
        for dir in [out_dir_base, out_dir_progress, out_dir_meta]:
            os.makedirs(dir, exist_ok=True)

        drawings = ndjson_drawings(cat)
        count = 0
        for d in drawings:
            if (n is not None) and (count == n):
                break

            id, strokes = d['key_id'], d['drawing']
            img = create_progression_image_from_ndjson_seq(strokes)

            # save
            img.save(out_dir_progress / f'{id}.jpg')

            # Save start and end strokes
            meta_fp = out_dir_meta / f'{id}.json'
            with open(meta_fp, 'w') as f:
                json.dump({'id': id, 'start': None, 'end': None, 'n_segments': len(strokes)}, f)

            count += 1

def save_progression_pairs(n=None):
    """Save two images in progression"""
    categories = final_categories()
    for cat in categories:
        print(cat)

        # make directories
        out_dir_base = QUICKDRAW_PROGRESSIONS_PAIRS_DATA_PATH
        out_dir_progress = out_dir_base / cat / 'progress'
        out_dir_meta = out_dir_base / cat / 'meta'
        for dir in [out_dir_base, out_dir_progress, out_dir_meta]:
            os.makedirs(dir, exist_ok=True)

        drawings = ndjson_drawings(cat)
        count = 0
        for d in drawings:
            if (n is not None) and (count == n):
                break

            id, strokes = d['key_id'], d['drawing']

            # this filters out a lot of incomplete drawings
            if len(strokes) < PAIRS_MIN_STROKES:
                continue

            # get image representations of strokes
            seg_vecs = []
            for s_idx, s in enumerate(strokes):
                segments = strokes[:s_idx+1]
                vec = vector_to_raster([segments], side=SIDE, line_diameter=LINE)[0]
                seg_vecs.append(vec.reshape(SIDE, SIDE))

            # insert blank
            seg_vecs.insert(0, np.zeros((SIDE, SIDE)))

            # randomly sample a segment
            # TODO: bias distribution to have shorter segments?
            last = len(seg_vecs) - 1
            start = random.randint(0, last - 1)
            end = random.randint(start + 1, last)

            #
            # Convert strokes to image and save
            #
            # create two images (each with border) and then paste together
            border = 3
            img1 = Image.fromarray(seg_vecs[start], 'L')
            img1 = ImageOps.invert(img1)
            img1 = ImageOps.expand(img1, border=border, fill='gray')
            img2 = Image.fromarray(seg_vecs[end], 'L')
            img2 = ImageOps.invert(img2)
            img2 = ImageOps.expand(img2, border=border, fill='gray')
            assert img1.size == img2.size
            img = Image.new('L', (SIDE * 2 + border * 3, SIDE + border * 2))
            img.paste(img1, (0, 0))
            img.paste(img2, (border + SIDE, 0))  # this way middle border overlaps
            img.save(out_dir_progress / f'{id}.jpg')

            # Save start and end strokes
            strokes_fp = out_dir_meta / f'{id}.json'
            with open(strokes_fp, 'w') as f:
                json.dump({'id': id, 'start': start, 'end': end, 'n_segments': len(strokes)}, f)

            count += 1

def prep_progressions_data_for_turk(data, n):
    """
    After manually filtering based on subsegments_v2/<base>/final, keep only the corresponding progressions.
    Write to csv with Amazon bucket name, category, start, end strokes in csv
    """
    data_dir, s3_url = None, None
    if data == 'progression_pairs':
        data_dir = QUICKDRAW_PROGRESSIONS_PAIRS_DATA_PATH
        s3_url = S3_PROGRESSION_PAIRS_URL
        out_fn = 'mturk_progressions_pairs_fullinput0.csv'
    elif data == 'progressions':
        data_dir = QUICKDRAW_PROGRESSIONS_PATH
        s3_url = S3_PROGRESSIONS_URL
        out_fn = 'mturk_progressions_fullinput0.csv'
    _prep_progressions_data_for_turk(data_dir, s3_url, out_fn, n)

def _prep_progressions_data_for_turk(data_dir, s3_url, out_fn, n):
    """
    Create the csv that will be input to MTurk

    :param data_dir: str, location of data to be annotated
    :param s3_url: str, url of image
    :param out_fn: str, filename of csv
    :param n: int, number of data points per category
    :return: None
    """
    categories = final_categories()
    csv_data = []
    for root, dirs, fns in os.walk(data_dir):
        category = os.path.basename(root)
        if category in categories:
            prog_dir = data_dir / category / 'progress'
            meta_dir = data_dir / category / 'meta'

            count = 0
            for fn in os.listdir(prog_dir):
                if n == count:
                    break
                if fn == '.DS_Store':
                    continue

                try:
                    # save data to csv
                    meta_fn = fn.replace('.jpg', '.json')
                    meta_fp = meta_dir / meta_fn
                    url = s3_url.format(category, fn)
                    meta = json.load(open(meta_fp, 'r'))
                    csv_data.append([category, url, str(meta['id']),
                                     str(meta['start']), str(meta['end']), str(meta['n_segments'])])
                except FileNotFoundError:
                    # some progressions were not saved, for example because they had too few strokes
                    pass

    # Write to csv
    csv_out_fp = data_dir / out_fn
    print(csv_out_fp)
    with open(csv_out_fp, 'w') as f:
        f.write('category,url,id,start,end,n_segments\n')
        random.shuffle(csv_data)
        for i, line in enumerate(csv_data):
            f.write(','.join(line) + '\n')

    # Calc stats
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 6})

    df = pd.read_csv(csv_out_fp)
    df.start = pd.to_numeric(df.start)
    df.end = pd.to_numeric(df.end)
    df.n_segments = pd.to_numeric(df.n_segments)
    df['diff'] = df.end - df.start
    df.start /= (df.n_segments + 1)
    df.end /= (df.n_segments + 1)

    out_dir = data_dir
    for col, color in [('start', 'green'),
                       ('end', 'red'),
                       ('diff', 'black'),
                       ('n_segments', 'orange')]:
        plt.figure(dpi=1200)
        df[col].hist(by=df.category, figsize=(10,8), alpha=0.8, color=color)
        # plt.suptitle('Distribution of {}'.format(col), fontsize=24)
        plt.tight_layout()
        out_fp = out_dir / f'{col}.png'
        plt.savefig(out_fp)



###################################################################
#
# AWS
#
###################################################################

def push_to_aws():
    AWS_PROFILE = 'ericchu56'
    AWS_CP_CMD = 'aws --profile {} s3 cp {} {}'

    categories = final_categories()
    for local_data_path, s3_path in [(QUICKDRAW_PROGRESSIONS_PAIRS_DATA_PATH, S3_PROGRESSION_PAIRS_PATH)]:
        for root, dirs, fns in os.walk(local_data_path):
            if os.path.basename(root) == 'progress':
                category = os.path.basename(os.path.dirname(root))
                if category in categories:
                    print(category)
                    for fn in fns:
                        local_fp = root / fn

                        s3_fp = s3_path.format(category, fn)
                        cp_cmd = AWS_CP_CMD.format(AWS_PROFILE, local_fp, s3_fp)
                        subprocess.Popen(cp_cmd.split())

                    sleep(45)

###################################################################
#
# Analyzing and saving annotated data
#
###################################################################

def convert_turk_results_to_html():
    """
    Convert csv from MTurk to a html file.
    """
    html_path = ANNOTATED_PROGRESSION_PAIRS_CSV_PATH.replace('.csv', '.html')
    with open(html_path, 'w') as out_f:
        out_f.write("""
        <html lang="en">
            <head>
              <title>Bootstrap Example</title>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
              <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
              <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
            </head>
            <body>

            <div class="container">
                <h2>MTurk Results</h2>
        """)

        ROW_TEMPLATE = """
        <div class="row">
            <div class="col-md-4">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>{}</p>
                  </div>
              </div>
            </div>
            <div class="col-md-4">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>{}</p>
                  </div>
              </div>
            </div>
            <div class="col-md-4">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>{}</p>
                  </div>
              </div>
            </div>
          </div>
        """

        df = pd.read_csv(ANNOTATED_PROGRESSION_PAIRS_CSV_PATH)
        for i in range(0, len(df) - (len(df) % 3), 3):
            row = ROW_TEMPLATE.format(
                df.iloc[i]['Input.category'],
                df.iloc[i]['Input.url'],
                df.iloc[i]['Answer.annotation'].replace('\r', '<br>'),

                df.iloc[i+1]['Input.category'],
                df.iloc[i+1]['Input.url'],
                df.iloc[i+1]['Answer.annotation'].replace('\r', '<br>'),

                df.iloc[i+2]['Input.category'],
                df.iloc[i+2]['Input.url'],
                df.iloc[i+2]['Answer.annotation'].replace('\r', '<br>'),
            )
            out_f.write(row)

        out_f.write("""
            </div>
            </body>
        </html>
        """)

def save_annotated_progression_pairs_data():
    """
    Save <category>.pkl files that is a dictionary from id to data.
    Data is a dictionary that contains:
        url: S3 url of progression pair
        annotation: instruction written by MTurker

        ndjson_strokes: drawing in ndjson format (list of subsegments, each subsegment is list of x y points)
        ndjson_start: ndjson_strokes index of start of annotated segment
            - Offset by 1 relative to ndjson_strokes
            - When 0, this is the start of the drawing (before any strokes)
        ndjson_end: ndjson_strokes index of end of annotated segment
            - Offset by 1 relative to ndjson_strokes

        stroke3: drawing in stroke-3 format: numpy array of shape [len, 3] (x, y, pen_up)
        stroke3_start: stroke3 index of start of annotated segment
        stroke3_end: stroke3 index of end of annotated segment
        stroke3_segment: numpy array of shape [len, 3] (x, y, pen_up)
            segment that was annotated (drawing from _start to _end of progression pair)
    """
    os.makedirs(LABELED_PROGRESSION_PAIRS_DATA_PATH, exist_ok=True)

    df = pd.read_csv(ANNOTATED_PROGRESSION_PAIRS_CSV_PATH)
    for cat in df['Input.category'].unique():
        df_cat = df[df['Input.category'] == cat]
        print(cat, len(df_cat))

        id_to_data = defaultdict(dict)

        # get ndjson stroke data
        drawings = ndjson_drawings(cat)
        id_to_strokes = defaultdict(dict)
        for data in drawings:
            id = data['key_id']
            id_to_strokes[id]['ndjson_strokes'] = data['drawing']
            stroke3 = ndjson_to_stroke3(data['drawing'])
            id_to_strokes[id]['stroke3'] = stroke3

        # map annotations to strokes
        for i in range(len(df_cat)):
            id = df_cat.iloc[i]['Input.id']
            id = str(id)
            annotation = df_cat.iloc[i]['Answer.annotation'].replace('\r', '')
            ndjson_start = df_cat.iloc[i]['Input.start']
            ndjson_end = df_cat.iloc[i]['Input.end']
            url = df_cat.iloc[i]['Input.url']

            id_to_data[id]['ndjson_start'] = int(ndjson_start)
            id_to_data[id]['ndjson_end'] = int(ndjson_end)
            id_to_data[id]['url'] = url
            id_to_data[id]['annotation'] = annotation

            ndjson_strokes = id_to_strokes[id]
            stroke3 = id_to_strokes[id]['stroke3']
            id_to_data[id]['ndjson_strokes'] =  id_to_strokes[id]['ndjson_strokes']
            id_to_data[id]['stroke3'] = stroke3

            # save portion of stroke3 corresponding to start and end
            pen_up = np.where(id_to_strokes[id]['stroke3'][:, 2] == 1)[0].tolist()
            pen_up.insert(0,0)  #  insert to get indexing (when ndjson_start == 0) this is the beginning
            stroke3_start = 0 if (ndjson_start == 0) else (pen_up[ndjson_start] + 1)
            stroke3_end = pen_up[ndjson_end]
            id_to_data[id]['stroke3_start'] = stroke3_start
            id_to_data[id]['stroke3_end'] = stroke3_end
            id_to_data[id]['stroke3_segment'] = stroke3[stroke3_start:stroke3_end+1, :]

        # flatten
        result = []
        for id, data in id_to_data.items():
            data['id'] = id
            data['category'] = cat
            result.append(data)

        # save
        out_fn = f'{cat}.pkl'
        out_fp = LABELED_PROGRESSION_PAIRS_DATA_PATH / out_fn
        utils.save_file(result, out_fp)


    # TODO: calculate tfidf on annotaitons

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--drawings', action='store_true')
    parser.add_argument('--progressions', action='store_true')
    parser.add_argument('--progression_pairs', action='store_true')
    parser.add_argument('--prep_progressions_data', action='store_true')
    parser.add_argument('--prep_data', type=str, default='progression_pairs',
                        help='"progressions" or "progression_pairs"')
    parser.add_argument('--prep_data_n', type=int, default=250, help='number of examples to get annotated')
    parser.add_argument('--push_to_aws', action='store_true')
    parser.add_argument('--html', action='store_true')
    parser.add_argument('--save_annotated_progression_pairs_data', action='store_true')
    args = parser.parse_args()

    if args.pairs:
        save_pairs(n=100)
    if args.drawings:
        save_final_drawings(n=1000)
    if args.progressions:
        save_progressions(n=1000)
    if args.progression_pairs:
        save_progression_pairs(n=250)
    if args.prep_progressions_data:
        prep_progressions_data_for_turk(args.prep_data, args.prep_data_n)
    if args.push_to_aws:
        push_to_aws()
    if args.html:
        convert_turk_results_to_html()
    if args.save_annotated_progression_pairs_data:
        save_annotated_progression_pairs_data()
