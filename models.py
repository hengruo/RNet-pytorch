import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg


class CharEmbedding(nn.Module):
    pass


class Embedding(nn.Module):
    pass


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.b_sz = cfg.batch_size
        self.bidirectional = cfg.enc_bidir
        self.dir = cfg.enc_dir
        self.num_layers = cfg.enc_num_layers
        self.hidden_size = cfg.hidden_size
        self.gru = nn.GRUCell(input_size=cfg.emb_size,
                          hidden_size=self.hidden_size, num_layers=self.num_layers,
                          bidirectional=self.bidirectional, dropout=cfg.dropout,
                          batch_first=cfg.batch_first)
        self.h = Variable(torch.randn(
            self.dir * self.num_layers, self.b_sz, self.hidden_size))

    def forward(self, input):
        (l, b_sz, in_sz) = input.size()
        assert(b_sz == self.b_sz)
        o, self.h = self.gru(input, self.h)
        return self.h


class PQMatcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()
        self.b_sz = cfg.batch_size
        self.hidden_size = cfg.hidden_size
        self.gru = nn.GRUCell(input_size=cfg.emb_size,
                          hidden_size=self.hidden_size)
        self.v = Variable(torch.randn(self.b_sz, self.hidden_size))
        self.Wp = nn.Linear(cfg.hidden_size * cfg.enc_num_layers * cfg.enc_dir, self.hidden_size)
        self.Wq = nn.Linear(cfg.hidden_size * cfg.enc_num_layers * cfg.enc_dir, self.hidden_size)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, Up, Uq):
        (lp, bz, sz) = Up.size()
        (lq, _, _) = Uq.size()
        Upr = Up[torch.arange(lp).long()]
        Uqr = Uq[torch.arange(lq).long()]
        assert(Upr.size() == (l, bz, sz))
        for i in range(l):
            # Wup.size() = (64, 75)
            Wup = self.Wp(Up[i])
            # Wuq.size() = (len, 64, 75)
            Wuq = self.Wq(Uq)
            # x.size() = (len, 64, 75)
            x = F.tanh(Wup + Wuq + self.Wv(self.v))
            # s.size() = (64, len, 1)
            s = torch.bmm(x.permute([1,0,2]), self.v.unsqueeze(2))
            # s.size() = (64, len)
            s = torch.squeeze(s, 2)

            # Wup.size() = (64, 75)
            Wup_ = self.Wp(Upr[i])
            Wuq_ = self.Wq(Uqr)
            x_ = F.tanh(Wup_ + Wuq_ + self.Wv(self.v))
            s_ = torch.bmm(x_.permute([1,0,2]), self.v.unsqueeze(2))
            s_ = torch.squeeze(s_, 2)

            a = F.softmax(s, 1)
            a_ = F.softmax(s_, 1)
            c = torch.bmm(a.unsqueeze(1) , Uq)


class SelfMatcher(nn.Module):
    pass

class Pointer(nn.Module):
    pass


class Predicter(nn.Module):
    pass


class RNet(nn.Module):
    def __init__(self):
        self.charemb = CharEmbedding()
        self.encoder = Encoder()
        self.pqmatcher = PQMatcher()

    # wemb of P, cemb of P, w of Q, c of Q, Answer
    def forward(self, Pw, Pc, Qw, Qc, A):
        (N, bz, _) = Pw.size()
        # Pcw.size() = (N, batch, 200)
        Pcw = self.charemb(Pc)
        Qcw = self.charemb(Qc)
        # P.size() = (N, batch, 500)
        P = torch.cat([Pw, Pcw], dim=2)
        Q = torch.cat([Qw, Qcw], dim=2)
        # Up.size() = (6, batch, 75)
        Up = self.encoder(P)
        Uq = self.encoder(Q)
        # Up.size() = (batch, 6, 75)
        Up = Up.permute([1, 0, 2])
        Uq = Uq.permute([1, 0, 2])
        # Up.size() = (batch, 450)
        Up = Up.view(bz, -1)
        Uq = Uq.view(bz, -1)
        #
        V = self.pqmatcher(Up, Uq)
