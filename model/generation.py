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


class Generator(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,  # equal to the dimension of the word vector, input_lang.n_words
                 num_layers=2,
                 bidirectional=False):
        super(Generator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dirs = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.decode = nn.Linear(self.dirs*hidden_size, output_size)  # take softmax
        
    def init_hidden(self, batch_size):
        h = Variable(torch.rand(self.dirs*self.num_layers, batch_size, self.hidden_size))
        return h
        
    def forward(self, x, hidden, seq_len=None, force=False):

        outputs = []
        if force:
            assert seq_len is None
            x, hidden = self.gru(x, hidden)  # x holds all of the sequence save for EOS
            for x_t in torch.unbind(x, 1):
                out = self.decode(x_t.view(x_t.size(0), -1))
                outputs.append(out)
        else:
            assert seq_len
            x_t = x  # x is only SOS
            for i in range(seq_len):
                x_t, hidden = self.gru(x_t, hidden)
                out = self.decode(x_t.view(x_t.size(0), -1))
                x_t = out.unsqueeze(1)
                outputs.append(out)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden


class Discriminator(nn.Module):
    def __init__(self, 
                 input_size,  # equal to input_lang.num_words
                 hidden_size=256,
                 num_layers=1,
                 bidirectional=False):
        
        super(Discriminator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dirs = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.output = nn.Sequential(
            nn.Linear(self.dirs*hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.dirs*self.num_layers, batch_size, self.hidden_size))
        return h
    
    def forward(self, x, hidden):

        outputs, hidden = self.gru(x, hidden)
        outputs = torch.unbind(outputs, 1)  # change this?

        out_class = self.output(outputs[-1].view(x.size(0), -1))
        
        return out_class, hidden
