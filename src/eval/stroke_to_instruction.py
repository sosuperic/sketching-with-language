# stroke_to_instruction.py

"""
Evaluating StrokeToInstruction model
"""
import argparse
from collections import defaultdict
import numpy as np
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
            score = sentence_bleu(utils.normalize_sentence(reference),
                                  utils.normalize_sentence(candidate))
            result = {'bleu': score}
        elif self.metric == 'rouge':
            result = {}
            for rouge_name, scores in self.scorer.score(reference, candidate).items():
                result[rouge_name] = scores.fmeasure
        return result


def calc_bleu_and_rouge_on_samples(samples_fp):
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
        for scorer in scorers:
            for name, value in scorer.score(gt, gen).items():
                m2scores[name].append(value)
                m2cat2scores[name][cat].append(value)

    print('\nROUGE and BLEU:')
    print('\nAverage per category:')
    for rouge_name, cat2scores in m2cat2scores.items():
        print('-' * 50)
        print(rouge_name)
        cat2avgs = {k: np.mean(v) for k, v in cat2scores.items()}
        pprint(sorted(cat2avgs.items(), key=lambda x: x[1]))

    print('Average:')
    pprint({rouge_name: np.mean(vals) for rouge_name, vals in m2scores.items()})

def calc_rare_words_stats(samples_fp):
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

    print('\nRare words stats:')
    print('Number of unique tokens in reference instructions: ', len(gt_toks))
    print('Number of unique tokens in generated instructions: ', len(gen_toks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', help='path to samples file')
    args = parser.parse_args()
    # calc_bleu_and_rouge_on_samples('samples_e11.json')
    # calc_bleu_and_rouge_on_samples('samples_eBEST.json')
    calc_bleu_and_rouge_on_samples(args.fp)
    calc_rare_words_stats(args.fp)
