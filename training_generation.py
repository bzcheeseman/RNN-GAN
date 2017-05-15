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

from model.generation import Generator, Discriminator, FeatureExtractor, InverseFeatureExtractor
from utils import *

use_cuda = False

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

embedding_dim = 128
gen_hidden_size = embedding_dim
batch = 1
max_length = 10

FE = FeatureExtractor(output_lang.n_words, embedding_dim)
InvFE = InverseFeatureExtractor(embedding_dim, output_lang.n_words)
G = Generator(
    hidden_size=embedding_dim
)
D = Discriminator(
    input_size=embedding_dim,
    hidden_size=256
)

FE_optimizer = optim.Adam(FE.parameters(), lr=1e-4, weight_decay=1e-6)
InvFE_optimizer = optim.Adam(InvFE.parameters(), lr=1e-4, weight_decay=1e-6)
G_optimizer = optim.Adam(G.parameters(), lr=1e-5)
D_optimizer = optim.Adam(D.parameters(), lr=1e-5)

FE_criterion = nn.CrossEntropyLoss()
D_criterion = nn.CrossEntropyLoss()

training_pairs = [variables_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(int(1e5))]

print_steps = 500

G.train()
G.cuda()

D.train()
# D.cuda()

real_target = Variable(torch.LongTensor([0]))
generated_target = Variable(torch.LongTensor([1]))

# g_train_range = (0.05, 0.5)
# d_lower_bound = 0.05
# train_g = False
# train_d = True

d_hidden = D.init_hidden(batch)
g_hidden = G.init_hidden(batch)

teacher_forcing_ratio = 0.2
ife_ratio = 1

G_running_loss = 0.0
D_running_loss = 0.0
IFE_running_loss = 0.0

for step in range(int(1e5)):  # gotta go through and check the detach() calls

    x = training_pairs[step][1]
    target = x[1:, :]
    gen_input = x[:-1, :]
    sos_token = x[0, :].unsqueeze(0)

    # Train D, feature extractor, inverse
    real_embedded = FE(target)  # embed the target
    real_prediction, d_hidden = D(real_embedded, d_hidden)
    loss_real = D_criterion(real_prediction, real_target)
    loss_real.backward()
    d_hidden = Variable(d_hidden.data)

    # Train D on fake data
    force = True if random.random() < teacher_forcing_ratio else False

    if force:
        g_input = FE(gen_input)
        fake_embedded, g_hidden = G(g_input.detach().cuda(), g_hidden, None, True)
        fake_prediction, d_hidden = D(fake_embedded.detach().cpu(), d_hidden)
        loss_fake = D_criterion(fake_prediction, generated_target)
        loss_fake.backward()
        d_hidden = Variable(d_hidden.data)
        g_hidden = Variable(g_hidden.data)
    else:
        g_input = FE(sos_token)
        fake_embedded, g_hidden = G(g_input.detach().cuda(), g_hidden, max_length, False)
        fake_prediction, d_hidden = D(fake_embedded.detach().cpu(), d_hidden)
        loss_fake = D_criterion(fake_prediction, generated_target)
        loss_fake.backward()
        d_hidden = Variable(d_hidden.data)
        g_hidden = Variable(g_hidden.data)

    D_running_loss += np.mean([loss_real.data[0], loss_fake.data[0]])
    FE_optimizer.step()
    D_optimizer.step()

    # Train inverse feature extractor
    for _i in range(ife_ratio):
        inverse_out = []
        for i, targ in enumerate(torch.unbind(target, 0), 0):
            inverse = InvFE(real_embedded[i].detach())
            inverse_out.append(Funct.softmax(inverse))
            loss_inv = FE_criterion(inverse, target[i])
            loss_inv.backward()
            IFE_running_loss += loss_inv.data[0]
        InvFE_optimizer.step()

    # Train G now
    if force:
        g_input = FE(gen_input)
        fake_embedded, g_hidden = G(g_input.detach().cuda(), g_hidden, None, True)  # we don't want to train FE
        gen_prediction, d_hidden = D(fake_embedded.cpu(), d_hidden)
        loss_gen = D_criterion(gen_prediction, real_target)  # G tries to make real-looking data
        loss_gen.backward()
        d_hidden = Variable(d_hidden.data)
        g_hidden = Variable(g_hidden.data)
    else:
        g_input = FE(sos_token)
        fake_embedded, g_hidden = G(g_input.detach().cuda(), g_hidden, max_length, False)  # don't train FE on G
        gen_prediction, d_hidden = D(fake_embedded.cpu(), d_hidden)
        loss_gen = D_criterion(gen_prediction, real_target)  # G tries to make real-looking data
        loss_gen.backward()
        d_hidden = Variable(d_hidden.data)
        g_hidden = Variable(g_hidden.data)

    G_running_loss += loss_fake.data[0]
    G_optimizer.step()

    if step % print_steps == print_steps-1:
        print('Iter-{} - D_loss: {:.4} - G_loss: {:.4} - IFE_loss: {:.4}'.format(
            step+1, D_running_loss/print_steps, G_running_loss/print_steps, IFE_running_loss/print_steps
        ))

        D.eval()
        G.eval()

        output_sentence_inverse = []
        for g_t in inverse_out:
            max, idx = g_t.data.topk(1)
            output_sentence_inverse.append(output_lang.index2word[idx[0, 0]])

        output_sentence = []
        for g_t in torch.unbind(target, 0):
            output_sentence.append(output_lang.index2word[g_t.data[0]])

        print("Real: {} - Inverse: {}".format(output_sentence, output_sentence_inverse))

        if IFE_running_loss/print_steps < 12.0:
            validate_gen(FE, G, InvFE, output_lang, sequence=np.random.randint(5, 10),
                         batch_size=1, sos_token=sos_token.cuda())

        D_running_loss = 0.0
        G_running_loss = 0.0
        IFE_running_loss = 0.0

        checkpoint(FE, "fe")
        checkpoint(InvFE, "ife")
        checkpoint(G, "g")
        checkpoint(D, "d")

        D.train()
        G.train()
