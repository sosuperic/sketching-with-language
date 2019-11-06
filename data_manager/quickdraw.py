"""
Analyzing and preparing QuickDraw data


https://github.com/googlecreativelab/quickdraw-dataset
"""

import argparse
from collections import defaultdict
import csv
import json
import math
import os
import random
from shutil import copyfile
import subprocess
from time import sleep

import cairocffi as cairo
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFont, ImageDraw


# Config and params
NDJSON_PATH = 'data/quickdraw/simplified_ndjson/{}.ndjson'
CATEGORIES_ANIMAL_PATH = 'data/quickdraw/categories_animals.txt'
CATEGORIES_FINAL_PATH = 'data/quickdraw/categories_final.txt'

QUICKDRAW_DATA_PATH = 'data/quickdraw'
QUICKDRAW_DRAWINGS_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'drawings')
QUICKDRAW_PAIRS_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'drawings_pairs')
QUICKDRAW_PROGRESSIONS_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'progressions')
FONT_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'ARIALBD.TTF')
QUICKDRAW_PROGRESSIONS_PAIRS_PATH = os.path.join(QUICKDRAW_DATA_PATH, 'progression_pairs_fullinput')

S3_PROGRESSIONS_URL = 'https://hierarchical-learning.s3.us-east-2.amazonaws.com/quickdraw/progressions_fullinput/{}/progress/{}'
S3_PROGRESSIONS_PATH = 's3://hierarchical-learning/quickdraw/progressions_fullinput/{}/progress/{}'
S3_PROGRESSION_PAIRS_URL = 'https://hierarchical-learning.s3.us-east-2.amazonaws.com/quickdraw/progression_pairs_fullinput/{}/progress/{}'
S3_PROGRESSION_PAIRS_PATH = 's3://hierarchical-learning/quickdraw/progression_pairs_fullinput/{}/progress/{}'

SIDE = 112
LINE = 6
PAIRS_MIN_STROKES = 3


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

###################################################################
#
# Saving and manipulating raw data
#
###################################################################

def save_pairs(n=None):
    """Save random pairs of drawings"""
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
            out_dir = os.path.join(QUICKDRAW_PAIRS_PATH, cat)
            os.makedirs(out_dir, exist_ok=True)
            out_fp = os.path.join(out_dir, '{}.jpg'.format(d_idx))
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
            out_dir = os.path.join(QUICKDRAW_DRAWINGS_PATH, cat)
            os.makedirs(out_dir, exist_ok=True)
            img.save(os.path.join(out_dir, '{}.jpg'.format(id)))

def save_progressions(n=None):
    """
    Saw progressions of strokes of data
    """
    font_size = int(SIDE * 0.2)
    font_space = 2 * font_size  # space for numbering
    font = ImageFont.truetype(FONT_PATH, font_size)
    segs_in_row = 8
    border = 3

    categories = animal_categories()
    for cat in categories:
        print(cat)

        # make directories
        out_dir_base = QUICKDRAW_PROGRESSIONS_PATH
        out_dir_progress = os.path.join(out_dir_base, cat, 'progress')
        out_dir_meta = os.path.join(out_dir_base, cat, 'meta')
        for dir in [out_dir_base, out_dir_progress, out_dir_meta]:
            os.makedirs(dir, exist_ok=True)

        drawings = ndjson_drawings(cat)
        count = 0
        for d in drawings:
            if (n is not None) and (count == n):
                break

            id, strokes = d['key_id'], d['drawing']

            n_segs = len(strokes)
            x_segs = n_segs if (n_segs < segs_in_row) else segs_in_row
            y_segs = math.ceil(n_segs / segs_in_row)
            x_height = SIDE * x_segs + border * (x_segs + 1)
            y_height = SIDE * y_segs + border * (y_segs + 1) + (font_space) * y_segs
            img = Image.new('L', (x_height, y_height))
            img.paste(255, [0,0,img.size[0], img.size[1]])  # fill in image with white
            for s_idx, s in enumerate(strokes):
                segments = strokes[:s_idx+1]
                vec = vector_to_raster([segments], side=SIDE, line_diameter=LINE)[0]
                seg_vec =  vec.reshape(SIDE, SIDE)

                seg_img = Image.fromarray(seg_vec, 'L')
                seg_img = ImageOps.expand(seg_img, (0, font_space, 0, 0))  # add white space above for number
                seg_img = ImageOps.invert(seg_img)
                seg_img = ImageOps.expand(seg_img, border=border, fill='gray')
                num_offset = 0.5 * font_size
                draw = ImageDraw.Draw(seg_img)
                draw.text((num_offset, num_offset), str(s_idx+1), (0), font=font)

                x_offset = (s_idx % segs_in_row) * (border + SIDE)
                y_offset = (s_idx // segs_in_row) * (border + font_space + SIDE)
                img.paste(seg_img, (x_offset, y_offset))

            # save
            img.save(os.path.join(out_dir_progress, '{}.jpg'.format(id)))

            # Save start and end strokes
            meta_fp = os.path.join(out_dir_meta, '{}.json'.format(id))
            with open(meta_fp, 'w') as f:
                json.dump({'id': id, 'start': None, 'end': None, 'n_segments': len(strokes)}, f)

            count += 1

def save_progression_pairs(n=None):
    """Save two images in progression"""
    categories = final_categories()
    for cat in categories:
        print(cat)

        # make directories
        out_dir_base = QUICKDRAW_PROGRESSIONS_PAIRS_PATH
        out_dir_progress = os.path.join(out_dir_base, cat, 'progress')
        out_dir_meta = os.path.join(out_dir_base, cat, 'meta')
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
            img.save(os.path.join(out_dir_progress, '{}.jpg'.format(id)))

            # Save start and end strokes
            strokes_fp = os.path.join(out_dir_meta, '{}.json'.format(id))
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
        data_dir = QUICKDRAW_PROGRESSIONS_PAIRS_PATH
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
            prog_dir = os.path.join(data_dir, category, 'progress')
            meta_dir = os.path.join(data_dir, category, 'meta')

            count = 0
            for fn in os.listdir(prog_dir):
                if n == count:
                    break
                if fn == '.DS_Store':
                    continue

                try:
                    # save data to csv
                    meta_fn = fn.replace('.jpg', '.json')
                    meta_fp = os.path.join(meta_dir, meta_fn)
                    url = s3_url.format(category, fn)
                    meta = json.load(open(meta_fp, 'r'))
                    csv_data.append([category, url, str(meta['id']),
                                     str(meta['start']), str(meta['end']), str(meta['n_segments'])])
                except FileNotFoundError:
                    # some progressions were not saved, for example because they had too few strokes
                    pass

    # Write to csv
    csv_out_fp = os.path.join(data_dir, out_fn)
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
        out_fp = os.path.join(out_dir, '{}.png'.format(col))
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
    for local_data_path, s3_path in [(QUICKDRAW_PROGRESSIONS_PAIRS_PATH, S3_PROGRESSION_PAIRS_PATH)]:
        for root, dirs, fns in os.walk(local_data_path):
            if os.path.basename(root) == 'progress':
                category = os.path.basename(os.path.dirname(root))
                if category in categories:
                    print(category)
                    for fn in fns:
                        local_fp = os.path.join(root, fn)

                        s3_fp = s3_path.format(category, fn)
                        cp_cmd = AWS_CP_CMD.format(AWS_PROFILE, local_fp, s3_fp)
                        subprocess.Popen(cp_cmd.split())

                    sleep(45)

###################################################################
#
# Analysis
#
###################################################################

def convert_turk_results_to_html():
    """Convert csv from MTurk to """
    RESULTS_PATH = os.path.join(QUICKDRAW_PROGRESSIONS_PAIRS_PATH, 'mturk_progressions_pairs_fullresults0.csv')
    HTML_PATH = os.path.join(QUICKDRAW_PROGRESSIONS_PAIRS_PATH, 'mturk_progressions_pairs_fullresults0.html')

    with open(HTML_PATH, 'w') as out_f:
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


        df = pd.read_csv(RESULTS_PATH)
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



if __name__ == '__main__':
    import argparse

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