'''
#################################
# Python API: ML Practice (Transformer Practice - MultiHead Self Attention)
#################################
'''

#########################################################
# import libraries
# from locale import nl_langinfo
import random
import numpy as np
import torch
import math
import copy
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections.abc import Sequence
from typing import Tuple, Optional, Union
# %matplotlib inline
# from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from copy import deepcopy

# from multihead import MultiHeadSelfAttention
# from attention.simpleattention import ScaledDotProductAttention
# from attention.multihead import MultiHeadSelfAttention

#########################################################
# General Parameters
#set seed to be able to replicate the resutls
seed = 172
random.seed(seed)
torch.manual_seed(seed)

#########################################################
# Function definition


def calc_degree_matrix(a):
        return torch.diag(a.sum(dim=-1))
    
    
def calc_graph_lapl(a):
    return calc_degree_matrix(a) - a


def test_degree():
    a = torch.rand(3, 3)
    a[a>0.5] = 1
    a[a<=0.5] = 0

    d = calc_degree_matrix(a)
    print('a', a)
    print('d', d)
    return a, d


def test_lapl():
    a = torch.rand(3, 3)
    a[a>0.5] = 1
    a[a<=0.5] = 0
    print('a', a)
    print('lapl', calc_graph_lapl(a))
    return calc_graph_lapl(a)
    

if __name__ == "__main__":
    test_lapl()