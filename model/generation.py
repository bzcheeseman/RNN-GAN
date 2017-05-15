#
# Created by Aman LaChapelle on 5/11/17.
#
# RNN-GAN
# Copyright (c) 2017 Aman LaChapelle
# Full license at RNN-GAN/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct


class FeatureExtractor(nn.Module):
    def __init__(self, num_words, embedding_dim):
        super(FeatureExtractor, self).__init__()

        self.fe = nn.Embedding(num_words, embedding_dim)

    def forward(self, x):
        return self.fe(x)


class InverseFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, num_words):
        super(InverseFeatureExtractor, self).__init__()

        self.inv_fe = nn.Linear(embedding_dim, num_words, bias=False)

    def forward(self, x):
        return Funct.relu(self.inv_fe(x))  # relu is ok?


class Generator(nn.Module):
    def __init__(self,
                 hidden_size,  # == embedding_dim in the feature extractor(s)
                 num_layers=2,
                 bidirectional=False):
        super(Generator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dirs = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional
        )

        self._gpu = False

    def cuda(self, device_id=0):
        self._gpu = True
        super(Generator, self).cuda(device_id)

    def init_hidden(self, batch_size):
        h = Variable(torch.rand(self.dirs*self.num_layers, batch_size, self.hidden_size))
        if self._gpu:
            return h.cuda()
        else:
            return h
        
    def forward(self, x, hidden, seq_len=None, force=False):

        if force:
            assert seq_len is None
            outputs, hidden = self.gru(x, hidden)  # x holds sequence except EOS, outputs should have all but SOS
        else:
            assert seq_len
            outputs = []
            x_t = x  # x is only SOS
            for i in range(seq_len):
                out, hidden = self.gru(x_t, hidden)
                outputs.append(out.squeeze(1))
                _, idx = out.data.topk(1)
                if idx[0, 0, 0] == 1:
                    break

            outputs = torch.stack(outputs, dim=0)

        return outputs, hidden


class Discriminator(nn.Module):
    def __init__(self, 
                 input_size,  # == embedding_dim
                 hidden_size=256,
                 num_layers=1,
                 bidirectional=True):
        
        super(Discriminator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dirs = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=bidirectional
        )
        self.output = nn.Sequential(
            nn.Linear(self.dirs*hidden_size, 64),
            nn.PReLU(),
            nn.Linear(64, 2)
        )
        
    def init_hidden(self, batch_size):
        h = Variable(torch.rand(self.dirs*self.num_layers, batch_size, self.hidden_size)*1e-3)  # give some noise
        return h
    
    def forward(self, x, hidden):

        outputs, hidden = self.gru(x, hidden)

        out_class = self.output(outputs[-1, :, :].view(x.size(1), -1))  # classify from last output
        
        return out_class, hidden
