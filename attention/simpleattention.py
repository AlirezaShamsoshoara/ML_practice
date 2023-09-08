'''
#################################
# Python API: ML Practice (GAN Practice)
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

class Attention(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, y_dim: int, h_dim: int):
        super().__init__()
        self.y_dim = y_dim
        self.h_dim = h_dim

        self.w = nn.Parameter(torch.FloatTensor(self.y_dim, self.h_dim))

    def forward(self, y: torch.Tensor, h: torch.Tensor):
        """_summary_

        Args:
            y (torch.Tensor): _description_
            h (torch.Tensor): _description_
        """
        e = torch.matmul(torch.matmul(y, self.w), h.T)
        score = F.softmax(e)
        attention = torch.matmul(score, h)
        return attention

class ScaledDotProductAttention(nn.Module):
    """_summary_
    class appears to implement the scaled dot-product attention mechanism, which is a crucial component of the transformer
    architecture used in deep learning, especially in tasks like natural language processing and sequence-to-sequence modeling.
    This mechanism calculates the attention scores between the query and key vectors and uses those scores to weight the values,
    allowing the model to focus on relevant parts of the input sequence. 
    Args:
        nn (_type_): _description_
    """
    def __init__(self, *args, **kwargs):
        super(ScaledDotProductAttention).__init__(*args, **kwargs)
        pass

    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None,
                droptout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, any]:
        """_summary_

        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len)
            `dropout`: nn.Dropout

        Returns:
            Tuple[torch.Tensor, any]: _description_
        """
        # Get the dimension of the key vectors.
        d_k = query.size(-1)

        # key.transpose(-2, -1) is a way to transpose a multi-dimensional array or tensor along its
        # two last dimensions.
        # For example, let's say you have a 4-dimensional tensor (a 4D array) key with shape
        # (batch_size, n_heads, max_len, d_k).
        # If you call key.transpose(-2, -1), it will swap the last two dimensions,
        # resulting in a new tensor with shape (batch_size, n_heads, d_k, max_len).
        # Calculate the dot products between query and key, and scale them.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if droptout is not None:
            p_attn = droptout(p_attn)

        # Compute the weighted sum of the values using the attention weights.
        output = torch.matmul(p_attn, value)
        return output, p_attn


if __name__ == "__main__":
    attention_model = Attention(y_dim=10, h_dim=20)
    y = torch.rand(1, 10)
    h = torch.rand(1, 20)
    print(attention_model(y=y, h=h))
    # train(model, training_data)
