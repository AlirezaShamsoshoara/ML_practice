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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1, *args, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(*args, **kwargs)
        assert d_model % n_heads == 0
        
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.linears = nn.ModuleList(
            [deepcopy(nn.Linear(in_features=d_model, out_features=d_model)) for _ in range(4)]
        )

    def forward(self, ):
        pass

if __name__ == "__main__":
    pass