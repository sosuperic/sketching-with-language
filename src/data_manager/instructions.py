# instructions.py

import argparse
from collections import defaultdict, Counter
import os
from pprint import pprint
import pandas as pd

import src.utils as utils
from src.data_manager.quickdraw import ANNOTATED_PROGRESSION_PAIRS_CSV_PATH, LABELED_PROGRESSION_PAIRS_PATH

INSTRUCTIONS_VOCAB_DISTRIBUTION_PATH = os.path.join(LABELED_PROGRESSION_PAIRS_PATH, 'vocab_distribution.json')



###################################################################
#
# Analyzing annotated data
#
###################################################################

def analyze_progression_pairs_annotations():
    df = pd.read_csv(ANNOTATED_PROGRESSION_PAIRS_CSV_PATH)

    words = Counter()
    for i in range(len(df)):
        id = df.iloc[i]['Input.id']
        annotation = df.iloc[i]['Answer.annotation'].replace('\r', '')
        for word in annotation.replace('.', '').lower().split():
            if word not in ['the', 'a', 'an', 'of']:
                words[word] += 1
    pprint(sorted(words.items(), key=lambda x: x[1]))

    # Count words by category
    for cat in df['Input.category'].unique():
        df_cat = df[df['Input.category'] == cat]
        words = Counter()
        for i in range(len(df_cat)):
            id = df_cat.iloc[i]['Input.id']
            annotation = df_cat.iloc[i]['Answer.annotation'].replace('\r', '')
            for word in annotation.replace('.', '').lower().split():
                if word not in ['the', 'a', 'an', 'of']:
                    words[word] += 1

        print('-' * 100)
        print
        print('CATEGORY: {}'.format(cat))
        pprint(sorted(words.items(), key=lambda x: x[1]))
        print('CATEGORY: {}'.format(cat))
        print

def save_instruction_vocabulary_distribution():

    df = pd.read_csv(ANNOTATED_PROGRESSION_PAIRS_CSV_PATH)

    tokens = Counter()
    for i in range(len(df)):
        annotation = df.iloc[i]['Answer.annotation'].replace('\r', '')
        for token in utils.normalize_sentence(annotation):
            tokens[token] += 1

    norm = sum(tokens.values())
    distribution = {tok: count / norm for tok, count in tokens.items()}
    utils.save_file(distribution, INSTRUCTIONS_VOCAB_DISTRIBUTION_PATH, verbose=True)


###################################################################
#
# Instruction Generation Outputs
#
###################################################################

def convert_generated_instruction_samples_to_html(samples_fp):
    """
    Convert outputs from StrokeToInstructionRNN model to html
    """
    html_path = samples_fp.replace('.json', '.html')
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
            <div class="col-md-5">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>Ground truth: {}</p>
                    <p>Generated: {}
                  </div>
              </div>
            </div>
            <div class="col-md-5">
              <div class="thumbnail">
                  <div>
                   <p><strong>Category: {}</strong></p>
                  </div>
                  <img src="{}" style="max-width:100%">
                  <div class="caption">
                    <p>Ground truth: {}</p>
                    <p>Generated: {}
                  </div>
              </div>
            </div>
          </div>
        """

        samples = utils.load_file(samples_fp)
        for i in range(0, len(samples), 2):
            # cat = sample['category']
            cat1 = samples[i]['url'].split('fullinput/')[1].split('/progress')[0]
            url1 = samples[i]['url']
            gt1 = ' '.join(utils.normalize_sentence(samples[i]['ground_truth']))
            gen1 = samples[i]['generated']

            cat2 = samples[i+1]['url'].split('fullinput/')[1].split('/progress')[0]
            url2 = samples[i+1]['url']
            gt2 = ' '.join(utils.normalize_sentence(samples[i+1]['ground_truth']))
            gen2 = samples[i+1]['generated']

            row = ROW_TEMPLATE.format(cat1, url1, gt1, gen1,
                                      cat2, url2, gt2, gen2)
            out_f.write(row)

        out_f.write("""
            </div>
            </body>
        </html>
        """)

if __name__ == '__main__':
    # samples_fp = 'samples_e11.json'
    # convert_generated_instruction_samples_to_html(samples_fp)

    # analyze_progression_pairs_annotations()

    save_instruction_vocabulary_distribution()