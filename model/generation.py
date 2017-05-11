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
                 num_layers,
                 output_size,  # equal to the dimension of the word vector, input_lang.n_words
                 bidirectional=False):
        super(Generator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dirs = 2 if bidirectional else 1
        
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            bidirectional=bidirectional
        )
        
        self.decode = nn.Linear(self.dirs*hidden_size, output_size)  # take softmax, torch.max, get indices
        
    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.dirs*self.num_layers, batch_size, self.hidden_size))
        return h
        
    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        indices = []
        for x_t in torch.unbind(x, 1):
            x_t = Funct.softmax(self.decode(x_t))
            _, idx = torch.max(x_t, 0)  # this might be wrong, worried about backprop working through here...
            indices.append(idx)
        indices = torch.stack(indices, dim=1)    
        return indices


class Discriminator(nn.Module):
    def __init__(self, 
                 input_size,  # equal to input_lang.num_words
                 hidden_size,
                 num_layers,
                 bidirectional=False):
        
        super(Discriminator, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dirs = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
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
        embedded = []
        for x_t in torch.unbind(x, 1):
            x_t = self.embedding(x_t)
            embedded.append(x_t)
        embedded = torch.stack(embedded, dim=1)
        
        _, hidden = self.gru(embedded, hidden)
        
        out_class = self.output(hidden.transpose(0, 1).view(x.size(0), -1))
        
        return out_class     
