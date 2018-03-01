import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import os
import ujson
import numpy as np
import random

def get_embedding(config):
    filename = os.path.join(config.embedding_root, config.embedding_name)
    f = open(filename, 'r')
    lines = f.readlines()
    vectors = torch.zeros(len(lines)+len(config.specials), config.embedding_dim)
    itos = config.specials
    stoi = defaultdict(lambda: len(stoi))
    stoi['<UNK>']
    stoi['<PAD>']
    stoi['<SOS>']
    stoi['<EOS>']
    for line in lines:
        tmp = line.strip().lower().split(' ')
        word = tmp[0]
        emb = [float(x) for x in tmp[1:]]
        itos.append(word)
        vectors[stoi[word], :] = torch.Tensor(emb)
    return itos, stoi, vectors

def parse_words(s):
    return s.split(' ')

def parse(s, wemb, cemb):
    words = parse_words(s)

def get_encoding(dir_path, filename, outfile, wemb, cemb):
    filename_ = os.path.join(dir_path, filename)
    with open(filename_, 'r') as f:
        data = ujson.load(f)
    data = data['data']
    cqas_list = []

    for topic in data:
        cqas = [{'context':      paragraph['context'],
                'id':           qa['id'],
                'question':     qa['question'],
                'answer':       qa['answers'][0]['text'],
                'answer_start': qa['answers'][0]['answer_start'],
                'answer_end':   qa['answers'][0]['answer_start'] + \
                                len(qa['answers'][0]['text']) - 1,
                'topic':        topic['title'] }
                for paragraph in topic['paragraphs']
                for qa in paragraph['qas']]

            cqas_list += cqas
    
    res = []
    for cqas in cqas_list:
        r = {}
        r['context'] = parse(cqas['context'], wemb, cemb)
        r['question'] = parse(cqas['question'], wemb, cemb)
        r['answer'] = parse(cqas['answer'], wemb, cemb)
        r['topic'] = parse(cqas['topic'], wemb, cemb)
        r['answer_start'] = cqas['answer_start']
        r['answer_end'] = cqas['answer_end']
    