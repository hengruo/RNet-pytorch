import argparse
from data import get_dataset
from data import SQuAD
from model import RNet
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg

def parse_args():
    args = argparse.ArgumentParser(description="An R-net implementation.")
    args.add_argument('--mode', dest='mode', type=str, default='all')
    args.add_argument("--batch_size", dest='batch_size', type=int, default="64")
    args.add_argument("--checkpoint", dest='checkpoint', type=int, default="1000")
    args.add_argument("--num_steps", dest='num_steps', type=int, default="60000")
    return args.parse_args()

def to_tensor(pack, data: SQuAD):
    # tensor representation of passage, question, answer
    # pw is a list of word embeddings.
    # pw[i] == data.word_embedding[data.train.wpassages[pack[0]]]
    pw, pc, qw, qc, a = [],[],[],[],[]
    return pw, pc, qw, qc, a

def train(data: SQuAD):
    model = RNet().cuda()
    for pack in data.train.pack:
        pw, pc, qw, qc, a = to_tensor(pack, data)
        output = model(pw, pc, qw, qc)
        loss = F.cross_entropy(output, a)
        optimizer = optim.Adadelta(loss)

def main():
    args = parse_args()
    data = get_dataset()
    train(data)


if __name__ == '__main__':
    main()