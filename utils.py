import array
import errno
import json
import os
import random
import zipfile
from argparse import ArgumentParser
from collections import Counter
from os.path import dirname, abspath

import nltk
import six
import torch
from six.moves.urllib.request import urlretrieve

URL = {
    'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip'
}


def reporthook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def load_word_vectors(root, wv_type, dim):
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        fname, _ = urlretrieve(url, fname)
        with zipfile.ZipFile(fname, "r") as zf:
            print('extracting word vectors into {}'.format(root))
            zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors %s from %s' % (wv_type, root))

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        print("Loading word vectors from {}".format(fname_txt))
        for line in trange(len(cm)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret


class RawExample(object):
    pass


def make_dirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def maybe_download(url, download_path, filename):
    if not os.path.exists(os.path.join(download_path, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            local_filename, _ = urlretrieve(url, os.path.join(download_path, filename))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e


def read_train_json(path, debug_mode, debug_len, delete_long_context=True, delete_long_question=True,
                    longest_context=300, longest_question=30):
    with open(path) as fin:
        data = json.load(fin)
    examples = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            passage = p['context']
            if delete_long_context and len(nltk.word_tokenize(passage)) > longest_context:
                continue
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]

                if delete_long_question and len(nltk.word_tokenize(question)) > longest_question:
                    continue

                question_id = qa["id"]
                for ans in answers:
                    answer_start = int(ans["answer_start"])
                    answer_text = ans["text"]
                    e = RawExample()
                    e.title = title
                    e.passage = passage
                    e.question = question
                    e.question_id = question_id
                    e.answer_start = answer_start
                    e.answer_text = answer_text
                    examples.append(e)

                    if debug_mode and len(examples) >= debug_len:
                        return examples
    print("train examples :%s" % len(examples))
    return examples


def get_counter(*seqs):
    word_counter = {}
    char_counter = {}
    for seq in seqs:
        for doc in seq:
            for word in doc:
                word_counter.setdefault(word, 0)
                word_counter[word] += 1
                for char in word:
                    char_counter.setdefault(char, 0)
                    char_counter[char] += 1
    return word_counter, char_counter


def read_dev_json(path, debug_mode, debug_len):
    with open(path) as fin:
        data = json.load(fin)
    examples = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']

            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                question_id = qa["id"]
                answer_start_list = [ans["answer_start"] for ans in answers]
                c = Counter(answer_start_list)
                most_common_answer, freq = c.most_common()[0]
                answer_text = None
                answer_start = None
                if freq > 1:
                    for i, ans_start in enumerate(answer_start_list):
                        if ans_start == most_common_answer:
                            answer_text = answers[i]["text"]
                            answer_start = answers[i]["answer_start"]
                            break
                else:
                    answer_text = answers[random.choice(range(len(answers)))]["text"]
                    answer_start = answers[random.choice(range(len(answers)))]["answer_start"]

                e = RawExample()
                e.title = title
                e.passage = context
                e.question = question
                e.question_id = question_id
                e.answer_start = answer_start
                e.answer_text = answer_text
                examples.append(e)

                if debug_mode and len(examples) >= debug_len:
                    return examples

    return examples


def tokenized_by_answer(context, answer_text, answer_start, tokenizer):
    fore = context[:answer_start]
    mid = context[answer_start: answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]

    tokenized_fore = tokenizer(fore)
    tokenized_mid = tokenizer(mid)
    tokenized_after = tokenizer(after)
    tokenized_text = tokenizer(answer_text)

    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None

    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token


def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}


def read_embedding(root, word_type, dim):
    wv_dict, wv_vectors, wv_size = load_word_vectors(root, word_type, dim)
    return wv_dict, wv_vectors, wv_size


def get_rnn(rnn_type):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        network = torch.nn.GRU
    elif rnn_type == "lstm":
        network = torch.nn.LSTM
    else:
        raise ValueError("Invalid RNN type %s" % rnn_type)
    return network


def sort_idx(seq):
    return sorted(range(seq.size(0)), key=lambda x: seq[x])


def prepare_data():
    make_dirs("data/cache")
    make_dirs("data/embedding/char")
    make_dirs("data/embedding/word")
    make_dirs("data/squad")
    make_dirs("data/trained_model")
    make_dirs("checkpoint")

    nltk.download("punkt")

    train_filename = "train-v1.1.json"
    dev_filename = "dev-v1.1.json"
    squad_base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

    train_url = os.path.join(squad_base_url, train_filename)
    dev_url = os.path.join(squad_base_url, dev_filename)

    download_prefix = os.path.join("data", "squad")
    maybe_download(train_url, download_prefix, train_filename)
    maybe_download(dev_url, download_prefix, dev_filename)

    char_embedding_pretrain_url = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt"
    char_embedding_filename = "glove_char.840B.300d.txt"
    maybe_download(char_embedding_pretrain_url, "data/embedding/char", char_embedding_filename)
