"Based on official script to obtain several statistical information"
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


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_question_type(dataset, predictions):
    # Which, When, What, Where, Why, How, Who, Whose, Other
    classify_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classify_f1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    classify_em = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question'].split()
                type = 8
                if question[0].lower() == 'which':
                    type = 0
                elif question[0].lower() == 'when':
                    type = 1
                elif question[0].lower() == 'what':
                    type = 2
                elif question[0].lower() == 'where':
                    type = 3
                elif question[0].lower() == 'why':
                    type = 4
                elif question[0].lower() == 'how':
                    type = 5
                elif question[0].lower() == 'who':
                    type = 6
                elif question[0].lower() == 'whose':
                    type = 7

                classify_count[type] += 1


                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]

                single_question_exact_match = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                single_question_f1 = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

                exact_match += single_question_exact_match
                f1 += single_question_f1

                classify_em[type] += single_question_exact_match
                classify_f1[type] += single_question_f1


    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    for i in range(9):
        classify_em[i] /= classify_count[i]
        classify_f1[i] /= classify_count[i]
        classify_em[i] *= 100
        classify_f1[i] *= 100

    print("Which, When, What, Where, Why, How, Who, Whose, Other")
    print("em: "+str(classify_em))
    print("f1: "+str(classify_f1))
    print("count: "+str(classify_count))

    return {'exact_match': exact_match, 'f1': f1}

def evaluate_len(dataset, predictions):
    max_context_length = 0
    max_question_length = 0
    max_answer_length = 0
    min_context_length = 100
    min_question_length = 100
    min_answer_length = 100
    total = exact_match = f1 = 0

    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context'].split()
            if len(context) > max_context_length:
                max_context_length = len(context)
            if len(context) < min_context_length:
                min_context_length = len(context)
            for qa in paragraph['qas']:
                question = qa['question'].split()
                if len(question) > max_question_length:
                    max_question_length = len(question)
                if len(question) < min_question_length:
                    min_question_length = len(question)

                for answers in qa['answers']:
                    answer = answers['text'].split()
                    if len(answer) > max_answer_length:
                        max_answer_length = len(answer)
                    if len(answer) < min_answer_length:
                        min_answer_length = len(answer)

    print([max_context_length,max_question_length,max_answer_length])
    print([min_context_length, min_question_length, min_answer_length])
    #
    # return {'exact_match': exact_match, 'f1': f1}

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
    # print(json.dumps(evaluate(dataset, predictions)))
