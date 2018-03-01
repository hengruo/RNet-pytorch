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
    filename = os.path.join(config.root, config.name)
    f = open(filename, 'r')
    lines = f.readlines()
    vectors = torch.zeros(len(lines)+len(config.specials), config.dim)
    itox = config.specials
    xtoi = defaultdict(lambda: len(xtoi))
    xtoi['<UNK>']
    xtoi['<PAD>']
    xtoi['<SOS>']
    xtoi['<EOS>']
    for line in lines:
        tmp = line.strip().lower().split(' ')
        word = tmp[0]
        emb = [float(x) for x in tmp[1:]]
        itox.append(word)
        vectors[xtoi[word], :] = torch.Tensor(emb)
    config['xtoi'] = xtoi
    config['itox'] = itox
    config['vec'] = vectors
    

def parse_words(s):
    return s.split(' ')

def parse_chars(s):
    return list(s)

def parse(s, wcf, ccf):
    words = parse_words(s)
    # chars = parse_chars(s)
    stoi = wcf['xtoi']
    wemb = wcf['vec']
    ctoi = ccf['xtoi']
    cemb = ccf['vec']
    word_ids = torch.Tensor([stoi(w) for w in words])
    wenc = wemb[word_ids, :]
    char_ids = torch.Tensor([ctoi(c) for c in s])
    cenc = cemb[char_ids, :]
    return wenc, cenc

def get_encoding(dir_path, filename, outfile, wcf, ccf):
    filename_ = os.path.join(dir_path, filename)
    with open(filename_, 'r') as f:
        data = ujson.load(f)
    data = data['data']
    cqas_list = []
    itop = []
    ptoi = defaultdict(lambda: len(ptoi))

    for topic in data:
        for paragraph in topic['paragraphs']:
            pid = ptoi[paragraph]
            itop.append(paragraph)
            for qa in paragraph['qas']:
                cqas = [{'context':     pid,
                        'id':           qa['id'],
                        'question':     qa['question'],
                        'answer':       qa['answers'][0]['text'],
                        'answer_start': qa['answers'][0]['answer_start'],
                        'answer_end':   qa['answers'][0]['answer_start'] + \
                                        len(qa['answers'][0]['text']) - 1,
                        'topic':        topic['title'] }]
                cqas_list += cqas
    
    res = []
    for cqas in cqas_list:
        r = {}
        r['context'] = parse(itop[cqas['context']], wcf, ccf)
        r['question'] = parse(cqas['question'], wcf, ccf)
        r['answer'] = parse(cqas['answer'], wcf, ccf)
        r['topic'] = parse(cqas['topic'], wcf, ccf)
        r['answer_start'] = cqas['answer_start']
        r['answer_end'] = cqas['answer_end']
        res.append(r)
    outfile_ = os.path.join(dir_path, outfile)
    with open(outfile_, 'w') as f:
        data = ujson.dump(res)
    f.close()
    