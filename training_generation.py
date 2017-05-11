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


def validate_gen(generator, lang, batch_size=1, sequence=10, gen_input=25):
    z = Variable(torch.rand(batch_size, sequence, gen_input))
    h = generator.init_hidden(batch_size)
    generated, h = generator(z.cuda(), h.cuda())

    # print out the sentence
    output_sentence = []
    for g_t in torch.unbind(generated.cpu(), 1):
        max, idx = torch.max(g_t, 1)
        output_sentence.append(lang.index2word[idx.data[0, 0]])
    print(" ".join(output_sentence))
    return generated.cpu()

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

input_size = output_lang.n_words
gen_input_size = 25
hidden_size = 128
batch = 1

to_onehot = IdxToOneHot(input_size)

G = Generator(gen_input_size, hidden_size, input_size)
D = Discriminator(input_size, hidden_size, 64)

G_optimizer = optim.Adam(G.parameters(), lr=5e-4, weight_decay=1e-3)
D_optimizer = optim.Adam(D.parameters(), lr=5e-5, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

training_pairs = [variables_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(int(1e5))]

h_G = G.init_hidden(batch)
h_D = D.init_hidden(batch)

running_loss_gen = 0.0
running_loss_disc = 0.0

print_steps = 100
g_ratio = 5

G.train()
G.cuda()

D.train()

validate_gen(G, output_lang)

for epoch in range(int(1e5)):

    x = training_pairs[epoch][1]
    x = x.unsqueeze(0).cpu()
    z = Variable(torch.rand(batch, x.size(1), gen_input_size))

    reset_grad([G, D])

    x = to_onehot(x.squeeze()).unsqueeze(0)

    # Train D
    output_real, h_D = D(x, h_D)
    target_real = Variable(torch.LongTensor([0]))
    loss_real = criterion(output_real, target_real)
    loss_real.backward()
    h_D = Variable(h_D.data)

    G_z, h_G = G(z.cuda(), h_G.cuda())
    target_fake = Variable(torch.LongTensor([1]))
    output_fake, h_D = D(G_z.cpu().detach(), h_D)  # don't train the generator on this part
    loss_fake = criterion(output_fake, target_fake)
    loss_fake.backward()
    h_D = Variable(h_D.data)
    h_G = Variable(h_G.data)

    D_optimizer.step()

    # Train G
    loss_gen = 0.0
    for _ in range(g_ratio):
        reset_grad([G, D])
        z = Variable(torch.rand(batch, x.size(1), gen_input_size))
        G_z, h_G = G(z.cuda(), h_G.cuda())
        target = Variable(torch.LongTensor([0]))
        output, h_D = D(G_z.cpu(), h_D)
        loss_gen = criterion(output, target)
        loss_gen.backward()
        h_D = Variable(h_D.data)
        h_G = Variable(h_G.data)

        G_optimizer.step()

    running_loss_gen += loss_gen.data[0]
    running_loss_disc += np.mean(loss_real.data[0] + loss_fake.data[0])

    if epoch % print_steps == print_steps-1:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(epoch+1, running_loss_disc/100, running_loss_gen/100))
        running_loss_disc = 0.0
        running_loss_gen = 0.0
        validate_gen(G, output_lang)
