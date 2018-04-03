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
batch_size = 32
hidden_size = 75
word_emb_size = 300
char_emb_size = char_dir * char_num_layers * char_hidden_size
emb_size = word_emb_size + char_emb_size

# Using bidirectional gru hidden state to represent char embedding for a word
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

# Input is the concatenation of word embedding and its corresponding char embedding
# Output is passage embedding or question embedding
class Encoder(nn.Module):
    def __init__(self, in_size, is_cuda):
        super(Encoder, self).__init__()
        self.bidirectional = True
        self.dir = 2 if self.bidirectional else 1
        self.num_layers = 3
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.gru = nn.GRU(input_size=in_size, bidirectional=self.bidirectional, num_layers=self.num_layers, hidden_size=self.hidden_size)
        self.out_size = self.hidden_size * self.num_layers * self.dir
        self.dropout = nn.Dropout(p=dropout)
        self.is_cuda = is_cuda

    def forward(self, input):
        (l, _, in_size) = input.size()
        hs = torch.zeros(l, self.num_layers * self.dir, batch_size, hidden_size)
        h = torch.randn(self.num_layers * self.dir, batch_size, self.hidden_size)
        if self.is_cuda:
            h = h.cuda()
            hs = hs.cuda()
        h = Variable(h)
        hs = Variable(hs)
        input = torch.unsqueeze(input, dim=1)
        for i in range(l):
            self.gru.flatten_parameters()
            o, h = self.gru(input[i], h)
            hs[i] = h
        del h
        hs = hs.permute([0,2,1,3]).contiguous().view(l, batch_size, -1)
        hs = self.dropout(hs)
        return hs

# Using passage and question to obtain question-aware passage representation
# Co-attention
class PQMatcher(nn.Module):
    def __init__(self, in_size, is_cuda):
        super(PQMatcher, self).__init__()
        self.hidden_size = hidden_size * 2
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size*2, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wq = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wg = nn.Linear(self.in_size*4, self.in_size*4, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.is_cuda = is_cuda

    
    def forward(self, up, uq):
        (lp, _, _) = up.size()
        (lq, _, _) = uq.size()
        mixerp, mixerq = torch.arange(lp).long(), torch.arange(lq).long()
        if self.is_cuda:
            mixerp, mixerq = mixerp.cuda(), mixerq.cuda()
        Up = torch.cat([up, up[mixerp]], dim=2)
        Uq = torch.cat([uq, uq[mixerq]], dim=2)
        vs = torch.zeros(lp, batch_size, self.out_size)
        v = torch.randn(batch_size, self.hidden_size)
        V = torch.randn(batch_size, self.hidden_size, 1)
        if self.is_cuda:
            vs = vs.cuda()
            v = v.cuda()
            V = V.cuda()
        vs = Variable(vs)
        v = Variable(v)
        V = Variable(V)
        for i in range(lp):
            Wup = self.Wp(Up[i])
            Wuq = self.Wq(Uq)
            x = F.tanh(Wup + Wuq + self.Wv(v))
            s = torch.bmm(x.permute([1, 0, 2]), V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1)
            c = torch.bmm(a.unsqueeze(1), Uq.permute([1, 0, 2])).squeeze()
            r = torch.cat([Up[i], c], dim=1)
            g = F.sigmoid(self.Wg(r))
            r = torch.mul(g, r)
            c_ = r[:, self.in_size*2:]
            v = self.gru(c_, v)
            vs[i] = v
        del v
        del V
        vs = self.dropout(vs)
        return vs.contiguous()

# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher(nn.Module):
    def __init__(self, in_size, is_cuda):
        super(SelfMatcher, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.is_cuda = is_cuda

    def forward(self, v):
        (l, _, _) = v.size()
        h = torch.randn(batch_size, self.hidden_size)
        V = torch.randn(batch_size, self.hidden_size, 1)
        hs = torch.zeros(l, batch_size, self.out_size)
        if self.is_cuda:
            h = h.cuda()
            V = V.cuda()
            hs = hs.cuda()
        h = Variable(h)
        V = Variable(V)
        hs = Variable(hs)
        for i in range(l):
            x = F.tanh(self.Wp(v[i])+self.Wp_(v))
            s = torch.bmm(x.permute([1, 0, 2]), V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1)
            c = torch.bmm(a.unsqueeze(1), v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
        del h
        del V
        hs = self.dropout(hs)
        return hs.contiguous()

# Input is question representation and self-attention question-aware passage representation
# Output are start and end pointer distribution
class Pointer(nn.Module):
    def __init__(self, in_size1, in_size2, is_cuda):
        super(Pointer, self).__init__()
        self.hidden_size = in_size2
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.gru = nn.GRUCell(input_size=in_size1, hidden_size=self.hidden_size)
        # Wu uses bias. See formula (11). Maybe Vr is just a bias.
        self.Wu = nn.Linear(self.in_size2, self.hidden_size, bias=True)
        self.Wh = nn.Linear(self.in_size1, self.hidden_size, bias=False)
        self.Wha = nn.Linear(self.in_size2, self.hidden_size, bias=False)
        self.out_size = 1
        self.is_cuda = is_cuda

    def forward(self, h, u):
        (lp, _, _) = h.size()
        (lq, _, _) = u.size()
        v = torch.randn(batch_size, self.hidden_size, 1)
        if self.is_cuda:
            v = v.cuda()
        v = Variable(v)
        x = F.tanh(self.Wu(u))
        s = torch.bmm(x.permute([1, 0, 2]), v)
        s = torch.squeeze(s, 2)
        a = F.softmax(s, 1)
        r = torch.bmm(a.unsqueeze(1), u.permute([1,0,2])).squeeze()
        x = F.tanh(self.Wh(h)+self.Wha(r))
        s = torch.bmm(x.permute([1, 0, 2]), v)
        s = torch.squeeze(s)
        p1 = F.softmax(s, 1)
        c = torch.bmm(p1.unsqueeze(1), h.permute([1,0,2])).squeeze()
        r = self.gru(c, r)
        x = F.tanh(self.Wh(h) + self.Wha(r))
        s = torch.bmm(x.permute([1, 0, 2]), v)
        s = torch.squeeze(s)
        p2 = F.softmax(s, 1)
        return (p1, p2)


class RNet(nn.Module):
    def __init__(self, is_cuda):
        super(RNet, self).__init__()
        self.encoder = Encoder(emb_size, is_cuda)
        self.pqmatcher = PQMatcher(self.encoder.out_size, is_cuda)
        self.selfmatcher = SelfMatcher(self.pqmatcher.out_size, is_cuda)
        self.pointer = Pointer(self.selfmatcher.out_size, self.encoder.out_size, is_cuda)
        self.is_cuda = is_cuda
        if is_cuda: self.cuda()

    # wemb of P, cemb of P, w of Q, c of Q, Answer
    def forward(self, Pw, Pc, Qw, Qc):
        lp = Pw.size()[0]
        lq = Qw.size()[0]
        P = torch.cat([Pw, Pc], dim=2)
        Q = torch.cat([Qw, Qc], dim=2)
        Up = self.encoder(P)
        Uq = self.encoder(Q)
        v = self.pqmatcher(Up, Uq)
        h = self.selfmatcher(v)
        p1, p2 = self.pointer(h, Uq)
        return p1, p2