import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config as cfg


class Encoder(nn.Module):
    def __init__(self, in_size, hidden_size, layer_num, 
                bidirec, dropout):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirec
        self.dropout = dropout
        super(Encoder, self).__init__()
        self.gru = nn.GRU(in_size, hidden_size, layer_num, 
            dropout=dropout, bidirectional=bidirec)
    
    def forward(self, in_data):
        d = 2 if self.bidirectional else 1
        seq_len, batch_size, _ = in_data.size()
        h0 = torch.randn(self.layer_num*d, batch_size, self.hidden_size)
        output, h = self.gru(in_data, h0)
        return output, h

class PQMatcher(nn.Module):
    '''
    Match a passage with an answer.
    '''
    def __init__(self, in_size, hidden_size, layer_num, 
                bidirec, dropout):
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirectional = bidirec
        self.dropout = dropout
        super(PQMatcher, self).__init__()
        self.lstm = nn.LSTM(in_size, hidden_size, layer_num, 
            dropout=dropout, bidirectional=bidirec)
        self.Wpu = nn.Linear(in_size, in_size, False)
        self.Wqu = nn.Linear(in_size, in_size, False)
        self.Wpv = nn.Linear(in_size, in_size, False)
    
    def forward(self, p_data, q_data):
        p_len, p_bsize, _ = p_data.size()
        q_len, q_bsize, _ = q_data.size()
        v = torch.randn(p_bsize, 1, self.in_size)
        for t in range(p_len):
            s = torch.zeros(q_len, p_bsize, 1)
            ut = p_data[t, :].view(p_bsize, 1, self.in_size)
            for j in range(q_len):
                uj = q_data[j, :].view(q_bsize, 1, self.in_size)
                sum = self.Wqu(uj) + self.Wpu(ut) + self.Wpv(v)
                s[j] = torch.bmm(v.view(p_bsize, self.in_size, 1), sum).view(p_bsize, -1)
            s = s.view(q_len, p_bsize)
            denom = torch.sum(s, 1)
            a = torch.div(s, denom)
            c = torch.zeros(p_bsize, 1, self.in_size)
            for i in range(q_len):
                ui = q_data[i, :].view(q_bsize, 1, self.in_size)
                ai = a[i, :]
                c.add_(torch.mul(ai, ui))
            v, _ = self.lstm(v, (ut, c))
        return v

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pencoder = Encoder(300, 2000, 3, True, 0)
        self.qencoder = Encoder(300, 2000, 3, True, 0)
        self.matcher = PQMatcher(300, 300, 1, False, 0)
        self.fc1 = nn.Linear(300, 2000)
        self.fc2 = nn.Linear(2000, 1500)
        self.fc3 = nn.Linear(1500, 1000)
        self.fc4 = nn.Linear(1000, 600)

    def forward(self, p_data, q_data):
        p = torch.cat(p_data, 1)
        q = torch.cat(q_data, 1)
        po, ph = self.pencoder(p)
        qo, qh = self.qencoder(q)
        v = self.matcher(po, qo)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        v = F.tanh(self.fc3(v))
        v = F.tanh(self.fc4(v))
        return v
