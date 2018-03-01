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
import pickle


def get_embedding(cfg):
    picklename = os.path.join(cfg['dir'], cfg['pkl_name'])
    if os.path.exists(picklename):
        with open(picklename, 'rb') as f:
            emb = pickle.load(f)
            cfg['xtoi'] = emb['xtoi']
            cfg['itox'] = emb['itox']
            cfg['vec'] = emb['vec']
    else:
        filename = os.path.join(cfg['dir'], cfg['raw_name'])
        with open(filename, 'r') as f:
            lines = f.readlines()
            vectors = torch.zeros(len(lines)+len(cfg['specials']), cfg['dim'])
            itox = cfg['specials']
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
            cfg['xtoi'] = xtoi
            cfg['itox'] = itox
            cfg['vec'] = vectors
            with open(picklename, 'wb') as ff:
                pickle.dump({'xtoi': cfg['xtoi'], 'itox': cfg['itox'], 'vec': cfg['vec']},
                            ff, pickle.HIGHEST_PROTOCOL)
                ff.close()


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


def get_encoding(cfg, wcf, ccf):
    picklename = os.path.join(cfg['dir'], cfg['pkl_name'])
    if os.path.exists(picklename):
        with open(picklename, 'rb') as f:
            cfg['data'] = pickle.load(f)
    else:
        filename = os.path.join(cfg['dir'], cfg['raw_name'])
        with open(filename, 'r') as f:
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
                                'answer_end':   qa['answers'][0]['answer_start'] +
                                len(qa['answers'][0]['text']) - 1,
                                'topic':        topic['title']}]
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
            cfg['data'] = res
            with open(picklename, 'w') as ff:
                pickle.dump(res, ff)
            ff.close()
