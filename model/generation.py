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
                 hidden_size,
                 output_size,  # equal to the dimension of the word vector, input_lang.n_words
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
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.decode = nn.Linear(self.dirs*hidden_size, output_size)  # take softmax, torch.max, get indices
        
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.dirs*self.num_layers, batch_size, self.hidden_size))
        return h
        
    def forward(self, x, hidden, seq_len):

        outputs = []
        x_t = x
        for i in range(seq_len):
            x_t, hidden = self.gru(x_t, hidden)

            out = Funct.softmax(self.decode(x_t.view(x_t.size(0), -1)))  # is this right..need to split into 2 loops?
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
        self.output = nn.Linear(self.num_layers*self.dirs*hidden_size, 2)
        
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.dirs*self.num_layers, batch_size, self.hidden_size))
        return h
    
    def forward(self, x, hidden):

        _, hidden = self.gru(x, hidden)
        
        out_class = self.output(hidden.transpose(0, 1).view(x.size(0), -1))
        
        return out_class, hidden
