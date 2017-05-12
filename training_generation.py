#
# Created by Aman LaChapelle on 5/11/17.
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
import numpy as np

from model.generation import Generator, Discriminator
from utils import *

use_cuda = False


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def validate_gen(generator, lang, sequence, batch_size=1, gen_input=128):
    z = Variable(torch.rand(batch_size, 1, gen_input))
    h = generator.init_hidden(batch_size)
    generated, h = generator(z.cuda(), h.cuda(), sequence)

    # print out the sentence
    output_sentence = []
    for g_t in torch.unbind(generated.cpu(), 1):
        max, idx = torch.max(g_t, 1)
        output_sentence.append(lang.index2word[idx.data[0, 0]])
    print(" ".join(output_sentence))
    return generated.cpu()

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

input_size = output_lang.n_words
gen_hidden_size = 250
disc_hidden_size = 128
gen_input_size = gen_hidden_size
batch = 1

to_onehot = IdxToOneHot(input_size)

G = Generator(hidden_size=gen_hidden_size, output_size=input_size, num_layers=2)
D = Discriminator(input_size=input_size, hidden_size=disc_hidden_size, bidirectional=False)

G_optimizer = optim.Adam(G.parameters(), lr=5e-5, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

criterion = nn.CrossEntropyLoss()

training_pairs = [variables_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(int(1e5))]

h_G = G.init_hidden(batch)
h_D = D.init_hidden(batch)

running_loss_gen = 0.0
running_loss_disc = 0.0

print_steps = 100
g_ratio = 3
d_ratio = 1

G.train()
G.cuda()

D.train()
D.cuda()

for epoch in range(int(1e5)):

    x = training_pairs[epoch][1]
    x = x.unsqueeze(0).cpu()

    x = to_onehot(x.squeeze()).unsqueeze(0)

    # Train D
    loss_real = 0.0
    loss_fake = 0.0
    for _i in range(d_ratio):
        reset_grad([G, D])
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        z = Variable(torch.rand(batch, 1, gen_input_size))
        output_real, h_D = D(x.cuda(), h_D.cuda())
        target_real = Variable(torch.LongTensor([0])).cuda()
        loss_real = criterion(output_real, target_real)
        loss_real.backward()
        h_D = Variable(h_D.data)

        G_z, h_G = G(z.cuda(), h_G.cuda(), x.size(1))
        target_fake = Variable(torch.LongTensor([1])).cuda()
        output_fake, h_D = D(G_z.detach(), h_D.cuda())  # don't train the generator on this part
        loss_fake = criterion(output_fake, target_fake)
        loss_fake.backward()
        h_D = Variable(h_D.data)
        h_G = Variable(h_G.data)

        D_optimizer.step()

    # Train G
    loss_gen = 0.0
    for _j in range(g_ratio):
        reset_grad([G, D])
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        z = Variable(torch.rand(batch, 1, gen_input_size))
        G_z, h_G = G(z.cuda(), h_G.cuda(), x.size(1))
        target = Variable(torch.LongTensor([0])).cuda()
        output, h_D = D(G_z.cuda(), h_D.cuda())
        loss_gen = criterion(output, target)
        loss_gen.backward()
        h_D = Variable(h_D.data)
        h_G = Variable(h_G.data)

        G_optimizer.step()

    running_loss_gen += loss_gen.data[0]
    running_loss_disc += np.mean(loss_real.data[0] + loss_fake.data[0])

    if epoch % print_steps == print_steps-1:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(epoch+1, running_loss_disc/print_steps, running_loss_gen/print_steps))

        running_loss_disc = 0.0
        running_loss_gen = 0.0

        output_sentence = []  # validate real data (just print to make sure it's not BS)
        for g_t in torch.unbind(x.cpu(), 1):
            max, idx = torch.max(g_t, 1)
            output_sentence.append(output_lang.index2word[idx.data[0, 0]])
        print(" ".join(output_sentence))

        validate_gen(G, output_lang, x.size(1), gen_input=gen_input_size)
