import argparse
import models
from dataset import get_dataset
from dataset import SQuAD
from models import RNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from tqdm import *
import os
import random
import ujson as uj
import eval

model_fn = "model.pt"
model_dir = "model/"
log_dir = "log/"
checkpoint = 1000
batch_size = 64
is_cuda = torch.cuda.is_available()


def parse_args():
    args = argparse.ArgumentParser(description="An R-net implementation.")
    args.add_argument('--mode', dest='mode', type=str, default='all')
    args.add_argument("--batch_size", dest='batch_size', type=int, default="64")
    args.add_argument("--checkpoint", dest='checkpoint', type=int, default="10000")
    args.add_argument("--epoch", dest='epoch', type=int, default="10")
    return args.parse_args()


def to_tensor(pack, data, dataset):
    # tensor representation of passage, question, answer
    # pw is a list of word embeddings.
    # pw[i] == data.word_embedding[dataset.wpassages[pack[0]]]
    assert batch_size == len(pack)
    Ps, Qs, As = [], [], []
    max_pl, max_ql = 0, 0
    for i in range(batch_size):
        pid, qid, aid = pack[i]
        p, q, a = dataset.passages[pid], dataset.questions[qid], dataset.answers[aid]
        max_pl, max_ql = max(max_pl, len(p)), max(max_ql, len(q))
        Ps.append(p)
        Qs.append(q)
        As.append(a)
    pw = torch.zeros(max_pl, batch_size, models.word_emb_size)
    pc = torch.zeros(max_pl, batch_size, models.char_emb_size)
    qw = torch.zeros(max_ql, batch_size, models.word_emb_size)
    qc = torch.zeros(max_ql, batch_size, models.char_emb_size)
    a = torch.zeros(batch_size, 2)
    for i in range(batch_size):
        pl, ql = len(Ps[i]), len(Qs[i])
        pw[:pl, i, :] = torch.FloatTensor(data.word_embedding[Ps[i]])
        pc[:pl, i, :] = torch.FloatTensor(data.char_embedding[Ps[i]])
        qw[:ql, i, :] = torch.FloatTensor(data.word_embedding[Qs[i]])
        qc[:ql, i, :] = torch.FloatTensor(data.char_embedding[Qs[i]])
        a[i, :] = torch.LongTensor(As[i])
    if is_cuda:
        pw, pc, qw, qc, a = Variable(pw.cuda()), Variable(pc.cuda()), Variable(qw.cuda()), Variable(qc.cuda()), Variable(a.cuda())
    else:
        pw, pc, qw, qc, a = Variable(pw), Variable(pc), Variable(qw), Variable(qc), Variable(a)
    return pw, pc, qw, qc, a.long()


def trunk(packs, batch_size):
    bpacks = []
    for i in range(0, len(packs), batch_size):
        bpacks.append(packs[i:i + batch_size])
    ll = len(bpacks[-1])
    if ll < batch_size:
        for j in range(batch_size - ll):
            bpacks[-1].append(random.choice(bpacks[-1]))
    random.shuffle(bpacks)
    return bpacks


def train(epoch, data):
    model = RNet()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-6)
    packs = trunk(data.train.packs, batch_size)
    try:
        for ep in range(epoch):
            print("EPOCH {:02d}: ".format(ep))
            l = data.train.length
            for i in tqdm(range(l)):
                pack = packs[i]
                pw, pc, qw, qc, a = to_tensor(pack, data, data.train)
                optimizer.zero_grad()
                out1, out2 = model(pw, pc, qw, qc)
                loss1 = F.cross_entropy(out1, a[:, 0])
                loss2 = F.cross_entropy(out2, a[:, 1])
                loss = (loss1 + loss2)/2
                loss.backward()
                optimizer.step()
                if (i + 1) % checkpoint == 0:
                    torch.save(model, os.path.join(model_dir, "model-tmp-{:02d}-{}.pt".format(ep, i + 1)))
        torch.save(model, os.path.join(model_dir, model_fn))
    except Exception as e:
        torch.save(model, os.path.join(model_dir, "model-{:02d}-{}.pt".format(ep, i + 1)))
        raise e
    return model

def get_anwser(i, j, pid, itow, dataset):
    p = dataset.passages[pid]
    i, j = max(i, j), min(i, j)
    ans_ = []
    for t in range(j-i+1):
        ans_.append(p[i+t])
    ans = ""
    for a in ans_:
        ans += itow[str(a)] + ' '
    return ans[:-1]

def test(model, data):
    l = data.dev.length
    packs = trunk(data.dev.packs, batch_size)
    err_cnt = 0
    anss = {}
    for i in tqdm(range(l)):
        pack = packs[i]
        pw, pc, qw, qc, a = to_tensor(pack, data, data.dev)
        out1, out2 = model(pw, pc, qw, qc)
        _, idx1 = torch.max(out1, dim=1)
        _, idx2 = torch.max(out2, dim=1)
        na = torch.cat([idx1.unsqueeze(1), idx2.unsqueeze(1)], dim=1)
        for j in range(batch_size):
            ans = get_anwser(na[j,0], na[j,1], pid, itow, dataset)
            anss[data.dev.question_ids[pack[j][1]]] = ans
    with open('log/answer.json', 'w') as f:
        uj.save(ans, f)
        em, f1 = eval.evaluate_from_file('tmp/dev-v1.1.json', 'log/answer.json')
        print("EM: {.3f}, F1: {.3f}".format(em, f1))

def main():
    args = parse_args()
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    data = get_dataset()
    if args.mode == 'all':
        model = train(args.epoch, data)
        test(model, data)
    elif args.mode == 'train':
        train(args.epoch, data)
    elif args.mode == 'test':
        model = torch.load(model_fn)
        test(model, data)
    else:
        print("Wrong arguments!")


if __name__ == '__main__':
    main()
