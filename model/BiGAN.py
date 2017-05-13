#
# Created by Aman LaChapelle on 5/10/17.
#
# RNN-GAN
# Copyright (c) 2017 Aman LaChapelle
# Full license at RNN-GAN/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers=2,
                 bidirectional=False,
                 use_cuda=True):

        super(Encoder, self).__init__()

        self.dirs = 2 if bidirectional else 1
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.n_layers*self.dirs, batch_size, self.hidden_size))
        if self.use_cuda:
            return h.cuda()
        else:
            return h

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        return x, hidden


class Generator(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers=1,
                 bidirectional=False,
                 use_cuda=True):

        super(Generator, self).__init__()

        self.dirs = 2 if bidirectional else 1
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        h = Variable(torch.rand(self.n_layers * self.dirs, batch_size, self.hidden_size))
        if self.use_cuda:
            return h.cuda()
        else:
            return h

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)

        outputs = []

        for x_t in torch.unbind(x, 1):
            x_t = Funct.softmax(self.linear(x_t))
            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=1)  # this should be [batch, seq, input_lang.n_words] as one-hot vectors

        return outputs, hidden


class Discriminator(nn.Module):
    def __init__(self,
                 data_dim,
                 feature_dim,
                 hidden_size,
                 n_layers=1,
                 bidirectional=False,
                 use_cuda=True):

        super(Discriminator, self).__init__()

        self.data_dim = data_dim
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dirs = 2 if bidirectional else 1
        self.use_cuda = use_cuda

        self.disc_gru = nn.GRU(
            input_size=feature_dim + data_dim,  # we've concatenated the embedded thing now
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.disc = nn.Sequential(
            nn.Linear(self.dirs * hidden_size, 2),
            # nn.PReLU(),
            # nn.Linear(64, 2)
        )

    def init_hidden(self, batch):
        h = Variable(torch.zeros(self.n_layers * self.dirs, batch, self.hidden_size))
        return h

    def forward(self, data, feature, hidden):
        
        x = torch.cat([data, feature], 2)
        
        outputs, hidden = self.disc_gru(x, hidden)
        outputs = torch.unbind(outputs, dim=1)
        x = self.disc(outputs[-1].view(outputs.size(0), -1))
        return x, hidden
