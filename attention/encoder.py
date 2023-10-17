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
from typing import Tuple, Optional, Union
# %matplotlib inline
# from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from copy import deepcopy

from attention.multihead import MultiHeadSelfAttention
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
"""_summary_
Linear layer as Feedforward, self attnetion part and skip connection with layer norm
"""

class FeedForward(nn.Module):
    
    """_summary_
    Args:
        nn (_type_): _description_
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, *args, **kwargs):
        """_summary_

        Args:
            d_model (int): _description_
            d_ff (int): _description_
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        super(FeedForward, self).__init__(*args, **kwargs)
        ## 1. DEFINE 2 LINEAR LAYERS AND DROPOUT HERE
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.feed_forward = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=128, ),
                nn.ReLU(),
                nn.Dropout(p = self.dropout),
                nn.Linear(in_features=128, out_features=d_ff),
                nn.Dropout(p = self.dropout),
        )
        # self.linear = nn.Linear(in_features=d_, out_features=, )

    def forward(self, x: torch.FloatTensor):
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
        """
        ## 2.  RETURN THE FORWARD PASS
        return self.feed_forward(x)


class LayerNorm(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, features: int, eps: float = 1e-6, *args, **kwargs):
        """_summary_

        Args:
            features (int): _description_
            eps (float, optional): _description_. Defaults to 1e-6.
        """
        super(LayerNorm, self).__init__(*args, **kwargs)
        self.a = nn.Parameter(torch.ones(features)) # Alpha parameter (scale)
        self.b = nn.Parameter(torch.zeros(features)) # Beta parameter (Shift)
        self.eps = eps

    def forward(self, x: torch.FloatTensor):
        """_summary_

        Args:
            x (torch.FloatTensor): _description_

        Returns:
            _type_: _description_
        """
        mean = x.mean(dim=-1, keepdim=True)
        # So, x.mean(-1, keepdim=True) calculates the mean along the last axis of the array x
        # and returns an array with the same number of dimensions as x, but with the mean 
        # values computed along the specified axis. This is particularly useful when you want
        # to maintain the shape of your array for further computations or broadcasting.
        std = x.std(dim=-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SkipConnection(nn.Module):
    """_summary_
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    Args:
        nn (_type_): _description_
    """
    def __init__(self, size: int, dropout: float, *args, **kwargs):
        """_summary_

        Args:
            size (int): _description_
            dropout (float): _description_
        """
        super(SkipConnection, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)

    def forward(self, x: torch.FloatTensor,
                sublayer: Union[MultiHeadSelfAttention, FeedForward]):
        """_summary_

        Args:
            x (torch.FloatTensor): _description_
            sublayer (Union[MultiHeadSelfAttention, FeedForward]): _description_

        Returns:
            _type_: _description_
        """
        return x + self.dropout(sublayer(self.norm(x)))

# class EncoderLayer(nn.Module):
#      """Encoder is made up of self-attn and feed forward"""
     
#      def __init__(self, size: int, self_attn: MultiHeadSelfAttention,
#                   feed_forward: FeedForward, dropout: float, *args, **kwargs) -> None:
#           super(EncoderLayer, self).__init__(*args, **kwargs)
#           self.self_attn = self_attn
#           self.dropout = dropout
#           self.feed_forward = feed_forward


if __name__ == "__main__":
    feed_forward = FeedForward(10, 100, 0.1)
    print(feed_forward(x=torch.rand(10)))
