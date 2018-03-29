""" Error analysis based on official evaluation script for the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, qa, paragraph, fcset):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, ""
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, ""


def exact_match_score(prediction, ground_truth, qa, paragraph, fcset):
    re = (normalize_answer(prediction) == normalize_answer(ground_truth))
    failcase = ""
    if not re and qa['id'] not in fcset:
        failcase += "\n" + qa['id'] + ": " + qa['question'] + "\n" + "prediction: " + prediction + "\n" + "ground_truth: " + ground_truth + "\n\n"
        fcset.add(qa['id'])
    return re, failcase

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, qa, paragraph, fcset):
    scores_for_ground_truths = []
    fcs = ""
    for ground_truth in ground_truths:
        score, failcase = metric_fn(prediction, ground_truth, qa, paragraph, fcset)
        scores_for_ground_truths.append(score)
        fcs += failcase
    return max(scores_for_ground_truths), fcs


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    fcset = set()
    with open("failcases.txt", 'w') as out_file:
        for article in dataset:
            for paragraph in article['paragraphs']:
                flag = False
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    em, failcase = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths, qa, paragraph, fcset)
                    if not em:
                        if not flag:
                            out_file.write("\n" + paragraph['context'] + "\n")
                            flag = True
                        out_file.write(failcase)
                    exact_match += em
                    ff1, sh1 = metric_max_over_ground_truths(
                        f1_score, prediction, ground_truths, qa, paragraph, fcset)
                    f1 += ff1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
