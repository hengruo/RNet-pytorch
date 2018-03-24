import os
import requests
import zipfile
import numpy as np
import ujson as uj
import spacy
from collections import defaultdict
import random

'''
Prepare data
'''

# configurations of data
train_filename = "train-v1.1.json"
dev_filename = "dev-v1.1.json"
char_emb_filename = "glove.840B.300d-char.txt"
word_emb_zip = "glove.840B.300d.zip"
word_emb_filename = "glove.840B.300d.txt"

data_dir = "tmp/squad"
emb_dir = "tmp/embedding"

emb_dim = 300

word_emb_url_base = "http://nlp.stanford.edu/data/"
char_emb_url_base = "https://raw.githubusercontent.com/minimaxir/char-embeddings/master/"
train_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
dev_url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"

# SQuAD dataset


class SQuAD:
    class Dataset:
        def __init__(self):
            self.length: int = 0
            self.wpassages: list = []
            self.wquestions: list = []
            self.cpassages: list = []
            self.cquestions: list = []
            self.answers: list(tuple) = [] # (start, end)
            self.pack: list(tuple) = [] # (pid, qid, aid)

    def __init__(self):
        self.word_embedding: np.ndarray = np.zeros((1, 1))
        self.char_embedding: np.ndarray = np.zeros((1, 1))
        self.train = SQuAD.Dataset()
        self.dev = SQuAD.Dataset()
        self.itow: dict = {}
        self.itoc: dict = {}
        self.ctoi: dict = {}
        self.wtoi: dict = {}
    
    @classmethod
    def load(cls, dirname: str):
        squad = SQuAD()
        f = open(os.path.join(dirname, "itow.json"), "r")
        squad.itow = uj.load(f)
        f.close()
        f = open(os.path.join(dirname, "itoc.json"), "r")
        squad.itoc = uj.load(f)
        f.close()
        f = open(os.path.join(dirname, "wtoi.json"), "r")
        squad.wtoi = uj.load(f)
        f.close()
        f = open(os.path.join(dirname, "ctoi.json"), "r")
        squad.ctoi = uj.load(f)
        f.close()
        f = open(os.path.join(dirname, "char_embedding.npy"), "rb")
        squad.char_embedding = np.load(f)
        f.close()
        f = open(os.path.join(dirname, "word_embedding.npy"), "rb")
        squad.word_embedding = np.load(f)
        f.close()
        f = open(os.path.join(dirname, "train.json"), "r")
        train = uj.load(f)
        for key in train:
            squad.train.__setattr__(key, train[key])
        f.close()
        f = open(os.path.join(dirname, "dev.json"), "r")
        dev = uj.load(f)
        for key in dev:
            squad.dev.__setattr__(key, train[key])
        f.close()
        return squad
    
    def dump(self, dirname: str):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        f = open(os.path.join(dirname, "itow.json"), "w")
        uj.dump(self.itow, f)
        f.close()
        f = open(os.path.join(dirname, "itoc.json"), "w")
        uj.dump(self.itoc, f)
        f.close()
        f = open(os.path.join(dirname, "wtoi.json"), "w")
        uj.dump(self.wtoi, f)
        f.close()
        f = open(os.path.join(dirname, "ctoi.json"), "w")
        uj.dump(self.ctoi, f)
        f.close()
        f = open(os.path.join(dirname, "char_embedding.npy"), "wb")
        np.save(f, self.char_embedding, allow_pickle=False)
        f.close()
        f = open(os.path.join(dirname, "word_embedding.npy"), "wb")
        np.save(f, self.word_embedding, allow_pickle=False)
        f.close()
        f = open(os.path.join(dirname, "train.json"), "w")
        uj.dump(self.train.__dict__, f)
        f.close()
        f = open(os.path.join(dirname, "dev.json"), "w")
        uj.dump(self.dev.__dict__, f)
        f.close()

def download(urlbase, filename, path):
    url = os.path.join(urlbase, filename)
    if not os.path.exists(os.path.join(path, filename)):
        try:
            print("Downloading file {}...".format(filename))
            r = requests.get(url, stream=True)
            fullname = os.path.join(path, filename)
            with open(fullname, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        except AttributeError as e:
            print("Download error!")
            raise e


def prepare_data():
    '''
    download squad dataset into `data/squad` and embedding into `data/embedding`
    '''
    dirs = [data_dir, emb_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
    download(train_url_base, train_filename, data_dir)
    download(dev_url_base, dev_filename, data_dir)
    download(char_emb_url_base, char_emb_filename, emb_dir)
    if not os.path.exists(os.path.join(emb_dir, word_emb_filename)):
        download(word_emb_url_base, word_emb_zip, emb_dir)
        zip_ref = zipfile.ZipFile(os.path.join(
            emb_dir, word_emb_zip), 'r')
        zip_ref.extractall(emb_dir)
        zip_ref.close()
        os.remove(os.path.join(emb_dir, word_emb_zip))


def parse_data_I(squad: SQuAD):
    cembf = open(os.path.join(emb_dir, char_emb_filename), 'r')
    wembf = open(os.path.join(emb_dir, word_emb_filename), 'r')
    ctoi = defaultdict(lambda: len(ctoi))
    wtoi = defaultdict(lambda: len(wtoi))
    specials = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
    [ctoi[x] for x in specials]
    [wtoi[x] for x in specials]
    cemb, wemb = [np.zeros(emb_dim) for _ in specials], [
        np.zeros(emb_dim) for _ in specials]
    for line in cembf.readlines():
        tmp = line.split(' ')
        ctoi[tmp[0]]
        cemb.append(np.array([float(x) for x in tmp[1:]]))

    for line in wembf.readlines():
        tmp = line.split(' ')
        wtoi[tmp[0]]
        wemb.append(np.array([float(x) for x in tmp[1:]]))
        if len(wtoi) != len(wemb):
            _ = wemb.pop()
            
    for char in ctoi:
        squad.itoc[ctoi[char]] = char
    for word in wtoi:
        squad.itow[wtoi[word]] = word
    squad.ctoi = dict(ctoi)
    squad.wtoi = dict(wtoi)
    squad.char_embedding = np.array(cemb)
    squad.word_embedding = np.array(wemb)

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def parse_dataset(jsondata, squad: SQuAD ,dataset: SQuAD.Dataset):
    tokenizer = spacy.blank('en')
    pid, qid, aid = 0, 0, 0
    for item in jsondata:
        for para in item['paragraphs']:
            context = para['context'].replace("''", '" ').replace("``", '" ')
            context_tokens = [token.text for token in tokenizer(context)]
            context_token_wids = [squad.wtoi[tk] if tk in squad.wtoi else 0 for tk in context_tokens]
            context_token_cids = [squad.ctoi[ch] if ch in squad.ctoi else 0 for ch in list(context)]
            spans = convert_idx(context, context_tokens)
            for qa in para['qas']:
                ques = qa['question'].replace("''", '" ').replace("``", '" ')
                ques_tokens = [token.text for token in tokenizer(ques)]
                ques_token_wids = [squad.wtoi[tk] if tk in squad.wtoi else 0 for tk in ques_tokens]
                ques_token_cids = [squad.ctoi[ch] if ch in squad.ctoi else 0 for ch in list(ques)]
                for ans in qa['answers']:
                    ans_text = ans['text']
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(ans_text)
                    ans_span = []
                    for idx, span in enumerate(spans):
                        if not (ans_end <= span[0] or ans_start >= span[1]):
                            ans_span.append(idx)
                    ans_pair = (ans_span[0], ans_span[-1])
                    dataset.answers.append(ans_pair)
                    dataset.pack.append((pid, qid, aid))
                    aid += 1
                dataset.wquestions.append(ques_token_wids)
                dataset.cquestions.append(ques_token_cids)
                qid += 1
            dataset.wpassages.append(context_token_wids)
            dataset.cpassages.append(context_token_cids)
            pid += 1
    dataset.length = len(dataset.pack)
    random.shuffle(dataset.pack)

def parse_data_II(squad):
    f = open(os.path.join(data_dir, train_filename), 'r')
    train = uj.load(f)['data']
    f.close()
    f = open(os.path.join(data_dir, dev_filename), 'r')
    dev = uj.load(f)['data']
    parse_dataset(train, squad, squad.train)
    parse_dataset(dev, squad, squad.dev)

def get_dataset():
    if os.path.exists('data/') and len(os.listdir('data/') == 8):
        squad = SQuAD.load('data/')
    else:
        prepare_data()
        squad = SQuAD()
        parse_data_I(squad)
        parse_data_II(squad)
        squad.dump("data/")
    return squad
