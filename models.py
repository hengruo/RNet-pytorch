import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

char_limit = 16
char_dim = 8
char_hidden_size = 100
char_num_layers = 1
char_dir = 2

dropout = 0.2
batch_size = 64
hidden_size = 75
word_emb_size = 300
char_emb_size = char_dir * char_num_layers * char_hidden_size
emb_size = word_emb_size + char_emb_size

class CharEmbedding(nn.Module):
    def __init__(self, in_size=word_emb_size):
        super(CharEmbedding, self).__init__()
        self.num_layers = 1
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.hidden_size = char_hidden_size
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers, hidden_size=self.hidden_size)
        self.h = Variable(torch.randn(self.num_layers*self.dir, 1, self.hidden_size))
        self.out_size = self.hidden_size * self.num_layers * self.dir

    def forward(self, input):
        (l, b, in_size) = input.size()
        assert in_size == self.in_size and b == 1
        o, h = self.gru(input, self.h)
        h = h.view(-1)
        return h

class Encoder(nn.Module):
    def __init__(self, in_size):
        super(Encoder, self).__init__()
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.num_layers = 3
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers, hidden_size=self.hidden_size)
        self.h = Variable(torch.randn(self.num_layers * self.dir, batch_size, self.hidden_size))
        self.out_size = self.hidden_size * self.num_layers * self.dir
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        (l, _, in_size) = input.size()
        hs = Variable(torch.zeros(l, self.num_layers*self.dir, batch_size, hidden_size))
        input = torch.unsqueeze(input, dim=1)
        for i in range(l):
            self.gru.flatten_parameters()
            o, self.h = self.gru(input[i], self.h)
            hs[i] = self.h
        hs = hs.permute([0,2,1,3]).contiguous().view(l, batch_size, -1)
        hs = self.dropout(hs)
        return hs


class PQMatcher(nn.Module):
    def __init__(self, in_size):
        super(PQMatcher, self).__init__()
        self.hidden_size = hidden_size * 2
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size*2, hidden_size=self.hidden_size)
        self.v = Variable(torch.randn(batch_size, self.hidden_size))
        self.V = Variable(torch.randn(batch_size, self.hidden_size, 1))
        self.Wp = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wq = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wg = nn.Linear(self.in_size*4, self.in_size*4, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, Up, Uq):
        (lp, _, _) = Up.size()
        (lq, _, _) = Uq.size()
        Up = torch.cat([Up, Up[torch.arange(lp).long()]], dim=2)
        Uq = torch.cat([Uq, Uq[torch.arange(lq).long()]], dim=2)
        vs = Variable(torch.zeros(lp, batch_size, self.out_size))
        for i in range(lp):
            Wup = self.Wp(Up[i])
            Wuq = self.Wq(Uq)
            x = F.tanh(Wup + Wuq + self.Wv(self.v))
            s = torch.bmm(x.permute([1, 0, 2]), self.V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1)
            c = torch.bmm(a.unsqueeze(1), Uq.permute([1, 0, 2])).squeeze()
            r = torch.cat([Up[i], c], dim=1)
            g = F.sigmoid(self.Wg(r))
            r = torch.mul(g, r)
            c_ = r[:, self.in_size*2:]
            self.gru.flatten_parameters()
            self.v = self.gru(c_, self.v)
            vs[i] = self.v
        vs = self.dropout(vs)
        return vs.contiguous()


class SelfMatcher(nn.Module):
    def __init__(self, in_size):
        super(SelfMatcher, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.h = Variable(torch.randn(batch_size, self.hidden_size))
        self.V = Variable(torch.randn(batch_size, self.hidden_size, 1))
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v):
        (l, _, _) = v.size()
        hs = Variable(torch.zeros(l, batch_size, self.out_size))
        for i in range(l):
            x = F.tanh(self.Wp(v[i])+self.Wp_(v))
            s = torch.bmm(x.permute([1, 0, 2]), self.V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1)
            c = torch.bmm(a.unsqueeze(1), v.permute([1, 0, 2])).squeeze()
            self.gru.flatten_parameters()
            self.h = self.gru(c, self.h)
            hs[i] = self.h
        hs = self.dropout(hs)
        return hs.contiguous()


class Pointer(nn.Module):
    def __init__(self, in_size1, in_size2):
        super(Pointer, self).__init__()
        self.hidden_size = in_size2
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.gru = nn.GRUCell(input_size=in_size1, hidden_size=self.hidden_size)
        self.v = Variable(torch.randn(batch_size, self.hidden_size, 1))
        # Wu uses bias. See formula (11). Maybe Vr is just a bias.
        self.Wu = nn.Linear(self.in_size2, self.hidden_size, bias=True)
        self.Wh = nn.Linear(self.in_size1, self.hidden_size, bias=False)
        self.Wha = nn.Linear(self.in_size2, self.hidden_size, bias=False)
        self.out_size = 1

    def forward(self, h, u):
        (lp, _, _) = h.size()
        (lq, _, _) = u.size()
        x = F.tanh(self.Wu(u))
        s = torch.bmm(x.permute([1, 0, 2]), self.v)
        s = torch.squeeze(s, 2)
        a = F.softmax(s, 1)
        r = torch.bmm(a.unsqueeze(1), u.permute([1,0,2])).squeeze()
        x = F.tanh(self.Wh(h)+self.Wha(r))
        s = torch.bmm(x.permute([1, 0, 2]), self.v)
        s = torch.squeeze(s)
        p1 = F.softmax(s, 1)
        c = torch.bmm(p1.unsqueeze(1), h.permute([1,0,2])).squeeze()
        self.gru.flatten_parameters()
        r = self.gru(c, r)
        x = F.tanh(self.Wh(h) + self.Wha(r))
        s = torch.bmm(x.permute([1, 0, 2]), self.v)
        s = torch.squeeze(s)
        p2 = F.softmax(s, 1)
        return (p1, p2)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.encoder = Encoder(emb_size)
        self.pqmatcher = PQMatcher(self.encoder.out_size)
        self.selfmatcher = SelfMatcher(self.pqmatcher.out_size)
        self.pointer = Pointer(self.selfmatcher.out_size, self.encoder.out_size)

    # wemb of P, cemb of P, w of Q, c of Q, Answer
    def forward(self, Pw, Pc, Qw, Qc):
        lp = Pw.size()[0]
        lq = Qw.size()[0]
        P = torch.cat([Pw, Pc], dim=2)
        Q = torch.cat([Qw, Qc], dim=2)
        assert P.size() == (lp, batch_size, emb_size)
        assert Q.size() == (lq, batch_size, emb_size)
        Up = self.encoder(P)
        Uq = self.encoder(Q)
        assert Up.size() == (lp, batch_size, self.encoder.out_size)
        assert Uq.size() == (lq, batch_size, self.encoder.out_size)
        v = self.pqmatcher(Up, Uq)
        assert v.size() == (lp, batch_size, self.pqmatcher.out_size)
        h = self.selfmatcher(v)
        assert h.size() == (lp, batch_size, self.selfmatcher.out_size)
        p1, p2 = self.pointer(h, Uq)
        return p1, p2