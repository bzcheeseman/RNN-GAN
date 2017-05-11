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


class Encoder(nn.Module):  # none of this is going to work properly
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers=1,
                 bidirectional=False,
                 use_cuda=True):
        super(Encoder, self).__init__()

        self.dirs = 2 if bidirectional else 1
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(self.n_layers*self.dirs, batch_size, self.hidden_size))  # make this learnable?
        if self.use_cuda:
            return h.cuda()
        else:
            return h

    def forward(self, x, hidden):
        embedded = []

        for x_t in torch.unbind(x, 1):
            x_t = x_t.unsqueeze(1).contiguous()
            x_t = self.embedding(x_t.view(x.size(0), -1))
            embedded.append(x_t)

        embedded = torch.stack(embedded, dim=1)

        x, hidden = self.gru(embedded.squeeze(2), hidden)
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
        h = Variable(torch.zeros(self.n_layers * self.dirs, batch_size, self.hidden_size))  # make this learnable?
        if self.use_cuda:
            return h.cuda()
        else:
            return h

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)

        indices = []

        for x_t in torch.unbind(x, 1):
            x_t = Funct.softmax(self.linear(x_t))
            _, idx = torch.max(x_t, 0)  # send out one-hot vectors for most likely word
            indices.append(idx)

        indices = torch.stack(indices, dim=1)

        return indices, hidden


class Discriminator(nn.Module):
    def __init__(self,
                 data_dim,
                 feature_dim,
                 hidden_size,
                 n_layers=1,
                 bidirectional=False
                 use_cuda=True):
        super(Discriminator, self).__init__()

        self.data_dim = data_dim
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dirs = 2 if bidirectional else 1
        self.use_cuda = use_cuda

        self.disc_input = data_dim + feature_dim
        self.enc = nn.Embedding(self.disc_input, hidden_size)
        self.disc_gru = nn.GRU(
            input_size=hidden_size,
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
        embedded = []
        for x_t in torch.unbind(x, 1):
            x_t = self.enc(x_t.view(x_t.size(0), -1))
            embedded.append(x_t)
        embedded = torch.stack(embedded, dim=1)
        
        _, hidden = self.disc_gru(embedded, hidden)
        x = self.disc(hidden.transpose(0, 1).view(hidden.size(0), -1))
        return x, hidden
