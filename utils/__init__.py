#
# Created by Aman LaChapelle on 5/10/17.
#
# RNN-GAN
# Copyright (c) 2017 Aman LaChapelle
# Full license at RNN-GAN/LICENSE.txt
#

# Taken from the pytorch seq2seq tutorials with minor changes - more changes to be made later on.

from io import open
import unicodedata
import string
import re
import random
import os

import torch
from torch.autograd import Variable

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


class IdxToOneHot:
    def __init__(self, total_num_indices):
        self.I = Variable(torch.eye(total_num_indices, total_num_indices))

    def forward(self, x):
        out = torch.index_select(self.I, 0, x)
        # add some noise - not enough to change anything (I don't think)  + Variable(torch.rand(o_t.size())*1e-5)
        out = torch.stack(
            [o_t for o_t in torch.unbind(out, 0)]
        )
        return out

    def __call__(self, input):
        return self.forward(input)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
         p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def get_sos_token():
    result = Variable(torch.LongTensor([SOS_token]).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.insert(0, SOS_token)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def variables_from_pair(input_lang, output_lang, pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return input_variable, target_variable


def validate_gen(fe, g, invfe, lang, sequence, batch_size, sos_token):
    h = g.init_hidden(batch_size)
    g_input = fe(sos_token)
    fake_embedded, h = g(g_input.detach(), h, sequence, False)

    # print out the sentence
    output_sentence = []
    for g_t in torch.unbind(fake_embedded.cpu(), 0):
        word = invfe(g_t)
        max, idx = word.data.topk(1)
        output_sentence.append(lang.index2word[idx[0, 0]])
    print("Generated: ", " ".join(output_sentence))


def checkpoint(net, name):
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(net.state_dict(), "checkpoints/ckpt_{}.pyt".format(name))

