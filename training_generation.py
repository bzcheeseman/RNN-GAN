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


def validate_gen(generator, lang, sequence, batch_size, z):
    h = generator.init_hidden(batch_size)
    generated, h = generator(z.cpu(), h.cpu(), sequence)

    # print out the sentence
    output_sentence = []
    for g_t in torch.unbind(generated.cpu(), 1):
        max, idx = torch.max(g_t, 1)
        output_sentence.append(lang.index2word[idx.data[0, 0]])
    print(" ".join(output_sentence))
    return generated.cpu()

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

input_size = output_lang.n_words
hidden_size = 128
batch = 1

to_onehot = IdxToOneHot(input_size)

G = Generator(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=input_size,
    num_layers=2,
    bidirectional=False
)
D = Discriminator(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=2,
    bidirectional=True
)

G_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

disc_criterion = nn.CrossEntropyLoss()
gen_criterion = nn.CrossEntropyLoss()

training_pairs = [variables_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(int(1e5))]

running_loss_gen = 0.0
running_loss_disc = 0.0

print_steps = 300
adjust_steps = 100
d_ratio = 1
g_ratio = 1

G.train()
# G.cuda()

D.train()
D.cuda()

generated_target = Variable(torch.LongTensor([0]))
forced_target = Variable(torch.LongTensor([1]))

g_train_range = (0.05, 0.5)
d_lower_bound = 0.05
train_g = False
train_d = True

for step in range(int(1e5)):

    x = training_pairs[step][1]
    target = x.unsqueeze(0)[:, 1:, :]
    x = x.unsqueeze(0).cpu()

    x = to_onehot(x.squeeze()).unsqueeze(0)
    start_token = x[:, 0, :].unsqueeze(1)
    x_force = x[:, :-1, :]
    # target = x[:, 1:, :]

    # output_sentence = []  # validate real data (just print to make sure it's not BS)
    # for g_t in torch.unbind(x_force.cpu(), 1):
    #     max, idx = torch.max(g_t, 1)
    #     output_sentence.append(output_lang.index2word[idx.data[0, 0]])
    # print(" ".join(output_sentence))
    #
    # output_sentence = []
    # for word in torch.unbind(target.cpu(), 1):
    #     output_sentence.append(output_lang.index2word[word.data[0, 0]])
    # print(" ".join(output_sentence))
    #
    # raise NameError

    reset_grad([G, D])
    G_optimizer.zero_grad()
    D_optimizer.zero_grad()

    for _i in range(d_ratio):
        # Run forced model
        h_force = G.init_hidden(batch)
        forced, h_force = G(x_force, h_force, None, True)

        # Run generative model
        h_gen = G.init_hidden(batch)
        generated, h_gen = G(start_token, h_gen, x.size(1)-1, False)

        # Train Discriminator on forced (both should have the same output)
        h_D = D.init_hidden(batch) + h_force.repeat(2, 1, 1)
        forced_d, h_D = D(forced.cuda().detach(), h_D.cuda())
        loss_disc_forced = disc_criterion(forced_d, forced_target.cuda())

        # Train Discriminator on generated (both should have the same output)
        h_D = D.init_hidden(batch) + h_gen.repeat(2, 1, 1)
        gen_d, h_D = D(generated.cuda().detach(), h_D.cuda())
        loss_disc_gen = disc_criterion(gen_d, generated_target.cuda())

        if train_d:
            loss_disc_forced.backward()
            loss_disc_gen.backward()
            D_optimizer.step()

    running_loss_disc += np.mean(loss_disc_forced.data[0] + loss_disc_gen.data[0])
    if 0.0 < np.mean(loss_disc_forced.data[0] + loss_disc_gen.data[0]) < d_lower_bound:
        train_d = False
    else:
        train_d = True

    for _j in range(g_ratio):
        # Train forced model (to get sequence right)
        h_force = G.init_hidden(batch)
        forced, h_force = G(x_force, h_force, None, True)
        loss_gen_forced = 0.0
        for i, t in enumerate(torch.unbind(forced, 1), 0):
            loss_gen_forced += gen_criterion(t, target[0, i, :])
        loss_gen_forced.backward()
        G_optimizer.step()

        if train_g:  # train generator to fool discriminator
            G_optimizer.zero_grad()
            D_optimizer.zero_grad()
            # Train generative model
            h_gen = G.init_hidden(batch)
            generated, h_gen = G(start_token, h_gen, x.size(1) - 1, False)

            # Train Discriminator on forced, don't need?
            # h_D = D.init_hidden(batch) + h_force.repeat(4, 1, 1)
            # forced_d, h_D = D(forced.cuda().detach(), h_D.cuda())
            # loss_forced = disc_criterion(forced_d, generated_target.cuda())  # swap targets
            # loss_forced.backward()

            # Train D and G on generated (both should have the same output)
            h_D = D.init_hidden(batch) + h_gen.repeat(2, 1, 1)
            gen_d, h_D = D(generated.cuda(), h_D.cuda())
            loss_generated = disc_criterion(gen_d, forced_target.cuda())  # swap targets, generator tries to fool disc
            loss_generated.backward()

            G_optimizer.step()
            # D_optimizer.step()  # do I need this?

    if train_g:
        running_loss_gen += np.mean(loss_gen_forced.data[0] + loss_generated.data[0])
    else:
        running_loss_gen += loss_gen_forced.data[0]

    # Adjust the ratio of D training to G training
    if g_train_range[0] <= np.mean(loss_disc_forced.data[0] + loss_disc_gen.data[0]) <= g_train_range[1]:
        train_g = True
    else:
        train_g = False

    if step % print_steps == print_steps-1:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; g/d: {}/{}'
              .format(step + 1, running_loss_disc / print_steps, running_loss_gen / print_steps, g_ratio, d_ratio))

        running_loss_disc = 0.0
        running_loss_gen = 0.0
        # g_ratio = 1
        # d_ratio = 2

        output_sentence = []  # validate real data (just print to make sure it's not BS)
        for g_t in torch.unbind(x[:, 1:, :].cpu(), 1):
            max, idx = torch.max(g_t, 1)
            output_sentence.append(output_lang.index2word[idx.data[0, 0]])
        print(" ".join(output_sentence))

        validate_gen(G, output_lang, np.random.randint(5, 10), 1, start_token)
