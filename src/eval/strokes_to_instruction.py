# strokes_to_instruction.py

"""
Evaluating StrokesToInstruction model by calcaulting ROUGE and other
word-based stats.

Usage:
    Specific samples path
        PYTHONPATH=. python src/eval/strokes_to_instruction.py --fp <dir>/outputs/samples_e11.json
    Big sweep
        PYTHONPATH=. python src/eval/strokes_to_instruction.py -d runs/strokes_to_instruction/dec18_2019/bigsweep/
"""

import argparse
from collections import defaultdict
import numpy as np
import os
from pprint import pprint

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

import src.utils as utils

class InstructionScorer(object):
    def __init__(self, metric):
        self.metric = metric

        if self.metric == 'rouge':
            # TODO: does this package do lower case, normalization
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def score(self, reference, candidate):
        """
        Args:
            reference: str
            candidate: str
        Returns:
            dict from metric_name (str) to score (float)
        """
        if self.metric == 'bleu':
            score1 = sentence_bleu([utils.normalize_sentence(reference)], utils.normalize_sentence(candidate), weights=[1.0])
            score2 = sentence_bleu([utils.normalize_sentence(reference)], utils.normalize_sentence(candidate), weights=[0.5, 0.5])
            result = {'bleu1': score1, 'bleu2': score2}
        elif self.metric == 'rouge':
            result = {}
            for rouge_name, scores in self.scorer.score(reference, candidate).items():
                result[rouge_name] = scores.fmeasure
        return result


def calc_bleu_and_rouge_on_samples(samples_fp, print=True):
    """
    Args:
        samples_fp: str to samples.json
    """
    samples = utils.load_file(samples_fp)

    scorers = [InstructionScorer('bleu'), InstructionScorer('rouge')]

    m2scores = defaultdict(list)
    m2cat2scores = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        # cat = sample['category']  # this wasn't saved in earlier runs.
        cat = sample['url'].split('fullinput/')[1].split('/progress')[0]
        gt, gen = sample['ground_truth'], sample['generated']
        # gt, gen = gt.lower(), gen.lower()
        # gt = gt.replace('draw', 'add')
        # gen = gen.replace('draw', 'add')
        for scorer in scorers:
            for name, value in scorer.score(gt, gen).items():
                m2scores[name].append(value)
                m2cat2scores[name][cat].append(value)

    if print:
        print('\nROUGE and BLEU:')
        print('\nAverage per category:')
        for rouge_name, cat2scores in m2cat2scores.items():
            print('-' * 50)
            print(rouge_name)
            cat2avgs = {k: np.mean(v) for k, v in cat2scores.items()}
            pprint(sorted(cat2avgs.items(), key=lambda x: x[1]))

        print('Average:')
        pprint({rouge_name: np.mean(vals) for rouge_name, vals in m2scores.items()})

    return m2scores, m2cat2scores

def calc_rare_words_stats(samples_fp, print=True):
    """
    Stats for whether rare words are being generated

    Args:
        samples_fp: str to samples.json
    """
    samples = utils.load_file(samples_fp)
    gt_toks = set()
    gen_toks = set()
    for sample in samples:
        gt, gen = sample['ground_truth'], sample['generated']
        for tok in utils.normalize_sentence(gt):
            gt_toks.add(tok)
        for tok in utils.normalize_sentence(gen):
            gen_toks.add(tok)

    if print:
        print('\nRare words stats:')
        print('Number of unique tokens in reference instructions: ', len(gt_toks))
        print('Number of unique tokens in generated instructions: ', len(gen_toks))

    return gt_toks, gen_toks

def calc_stats_for_runs_in_dir(dir, best_n=10):
    """
    Print runs with best stats in <dir>
    Assumes each run has file with the name: 'e<epoch>_loss<loss>.pt'.

    Args:
        dir (str)
        best_n (int)
    """
    print(f'Looking in: {dir}\n')

    n = 0
    runs_stats = []
    hiplot_stats = []
    for root, dirs, fns in os.walk(dir):
        for fn in fns:
            # Get loss from model fn (e<epoch>_loss<loss>.pt)
            if fn.endswith('pt') and ('loss' in fn):
                # Get best samples
                epoch = fn.split('_')[0].replace('e', '')
                loss = float(fn.split('loss')[1].strip('.pt'))
                run = root.replace(dir + '/', '')
                best_sample_fp = os.path.join(root, 'outputs', f'samples_e{epoch}.json')

                # Calculate stats
                m2scores, m2cat2scores = calc_bleu_and_rouge_on_samples(best_sample_fp, print=False)
                gt_toks, gen_toks = calc_rare_words_stats(best_sample_fp, print=False)
                run_results = {
                    'n_gen_toks': len(gen_toks),
                    'n_gt_toks': len(gt_toks),
                    'loss': loss,
                    'rouge1': np.mean(m2scores['rouge1']),
                    'rouge2': np.mean(m2scores['rouge2']),
                    'rougeL': np.mean(m2scores['rougeL']),
                    'bleu1': np.mean(m2scores['bleu1']),
                    'bleu2': np.mean(m2scores['bleu2']),
                }
                runs_stats.append([run, run_results])

                # Save json data to be visualized by hiplot
                hp_dict = utils.load_file(os.path.join(root, 'hp.json'))
                run_hiplot = {}
                for k, val in hp_dict.items():
                    run_hiplot[k] = val
                run_hiplot.update(run_results)
                hiplot_stats.append(run_hiplot)

                n += 1

    #
    # Write best runs sorted to file
    out_fp = os.path.join(dir, 'best_runs.txt')
    with open(out_fp, 'w') as f:
        print('-' * 100)
        for main_stat in runs_stats[0][1].keys():  # n_gen_toks, loss, rougeL, bleu1, bleu2
            print(f'RUNS WITH BEST: {main_stat}', file=f)
            if main_stat == 'loss':  # lower is beter
                sorted_by_main_stat = sorted(runs_stats, key=lambda x: -x[1][main_stat])[-best_n:]
            else:  # higher is better
                sorted_by_main_stat = sorted(runs_stats, key=lambda x: x[1][main_stat])[-best_n:]

            for run, stats in sorted_by_main_stat:
                main_stat_val = stats[main_stat]
                other_stats_str = ', '.join(['{}: {:.4f}'.format(stat, val) for stat, val in stats.items() if (main_stat != stat)])
                out_str = '{}: {:.4f}'.format(main_stat, main_stat_val)
                print(out_str + ', ' + other_stats_str + ', run: ' + run, file=f)
            print(file=f)

    # Print to stdout
    for line in open(out_fp, 'r').readlines():
        print(line.strip())
    print('\nWrote best runs sorted to: ', out_fp)

    #
    # Save hiplot data in runs/strokes_to_instruction/Feb14_2020/imagesweep_textaug_rankimgs/
    #
    out_fn = 'hiplot_data.json'
    out_fp = os.path.join(dir, out_fn)
    print()
    utils.save_file(hiplot_stats, out_fp, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', help='path to samples file')
    parser.add_argument('-d', '--dir', help='calculate stats for every run within this directory')
    args = parser.parse_args()

    if args.fp:
        calc_bleu_and_rouge_on_samples(args.fp)
        calc_rare_words_stats(args.fp)
    if args.dir:
        calc_stats_for_runs_in_dir(args.dir)
