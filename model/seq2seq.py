#
# Created by Aman LaChapelle on 5/10/17.
#
# RNN-GAN
# Copyright (c) 2017 Aman LaChapelle
# Full license at RNN-GAN/LICENSE.txt
#

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

# use chain() from itertools to combine parameters from multiple networks in one optimizer
# use zip() to combine two lists into a list of tuples

# Prof forcing - https://arxiv.org/pdf/1610.09038.pdf (USE/DO THIS)

# improve GAN training https://arxiv.org/pdf/1606.03498.pdf

# see https://arxiv.org/pdf/1611.09904.pdf C-RNN-GAN

# see TextGAN, ali_bigan (both starred on github),
#   BiGAN: https://arxiv.org/pdf/1605.09782.pdf, https://arxiv.org/pdf/1606.00704.pdf
#   language independent feature extraction with bigan - maybe even a parameter to select a language

# TGAN https://arxiv.org/pdf/1611.06624.pdf

# StackGAN https://arxiv.org/pdf/1612.03242.pdf

# CoGAN https://arxiv.org/pdf/1606.07536.pdf

# DualGAN https://arxiv.org/pdf/1704.02510.pdf

# InfoGAN https://arxiv.org/pdf/1606.03657.pdf


