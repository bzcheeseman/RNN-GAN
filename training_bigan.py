#
# Created by Aman LaChapelle on 5/10/17.
#
# RNN-GAN
# Copyright (c) 2017 Aman LaChapelle
# Full license at RNN-GAN/LICENSE.txt
#

import random
from itertools import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as Funct

from utils import *
from model import Encoder, Generator, Discriminator

# This still doesn't work

use_cuda = False
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

input_size = input_lang.n_words
hidden_size = 256
batch = 1

to_onehot = IdxToOneHot(input_size)

Q = Encoder(input_size, hidden_size, use_cuda=use_cuda)
P = Generator(hidden_size, hidden_size, input_size, use_cuda=use_cuda)
D = Discriminator(input_size, hidden_size, 64, use_cuda=use_cuda)

G_optimizer = optim.Adam(chain(Q.parameters(), P.parameters()), lr=5e-5, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

criterion = nn.CrossEntropyLoss()

training_pairs = [variables_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(int(1e5))]

h_Q = Q.init_hidden(batch)
h_P = P.init_hidden(batch)
h_D_enc = D.init_hidden(batch)
h_D_gen = D.init_hidden(batch)

for epoch in range(int(1e5)):

    x = training_pairs[epoch][0]
    x = x.unsqueeze(0).cpu()

    reset_grad([Q, P, D])

    x = to_onehot(x.squeeze()).unsqueeze(0)

    z = Variable(torch.zeros(batch, x.size(1), hidden_size))
    E_x, h_E = Q(x.cpu(), h_Q.cpu())
    G_z, h_G = P(z.cpu(), h_P.cpu())

    D_enc, h_D_enc = D(x, E_x, h_D_enc)
    # target_enc = Variable(torch.LongTensor([0]))
    z = Variable(torch.zeros(batch, x.size(1), hidden_size))
    D_gen, h_D_gen = D(G_z, z, h_D_gen)
    # target_gen = Variable(torch.LongTensor([1]))
    D_enc = Funct.sigmoid(D_enc)
    D_gen = Funct.sigmoid(D_gen)

    D_loss = -torch.mean(torch.log(D_enc) + torch.log(1 - D_gen))

    D_loss.backward(retain_variables=True)
    D_optimizer.step()
    G_optimizer.step()

    h_Q = Variable(h_Q.data)
    h_P = Variable(h_P.data)
    h_D_enc = Variable(h_D_enc.data)
    h_D_gen = Variable(h_D_gen.data)
    reset_grad([Q, P, D])

    z = Variable(torch.zeros(batch, x.size(1), hidden_size))
    E_x, h_Q = Q(x.cpu(), h_Q.cpu())
    G_z, h_P = P(z.cpu(), h_P.cpu())

    D_enc, h_D_enc = D(x, E_x, h_D_enc)
    # target_enc = Variable(torch.LongTensor([0]))
    z = Variable(torch.zeros(batch, x.size(1), hidden_size))
    D_gen, h_D_gen = D(G_z, z, h_D_gen)
    # target_gen = Variable(torch.LongTensor([1]))
    D_enc = Funct.sigmoid(D_enc)
    D_gen = Funct.sigmoid(D_gen)

    G_loss = -torch.mean(torch.log(D_gen) + torch.log(1 - D_enc))

    G_loss.backward()
    G_optimizer.step()

    h_Q = Variable(h_Q.data)
    h_P = Variable(h_P.data)
    h_D_enc = Variable(h_D_enc.data)
    h_D_gen = Variable(h_D_gen.data)
    reset_grad([Q, P, D])

    if epoch % 100 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(epoch, D_loss.data[0], G_loss.data[0]))
