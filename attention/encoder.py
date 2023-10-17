'''
#################################
# Python API: ML Practice (Transformer Practice - MultiHead Self Attention)
#################################
'''

#########################################################
# import libraries
import random
import numpy as np
import torch
import math
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections.abc import Sequence
from typing import Tuple, Optional
# %matplotlib inline
# from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from copy import deepcopy
from attention.simpleattention import ScaledDotProductAttention
from attention.multihead import MultiHeadSelfAttention

#########################################################
# General Parameters
#set seed to be able to replicate the resutls
seed = 172
random.seed(seed)
torch.manual_seed(seed)

#########################################################
# Function definition
"""_summary_

"""


class EncoderLayer(nn.Module):
     """Encoder is made up of self-attn and feed forward"""
     
     def __init__(self, size: int, self_attn: MultiHeadSelfAttention,
                  feed_forward: FeedForward, dropout: float, *args, **kwargs) -> None:
          super(EncoderLayer, self).__init__(*args, **kwargs)
          self.self_attn = self_attn
          self.dropout = dropout
          self.feed_forward = feed_forward