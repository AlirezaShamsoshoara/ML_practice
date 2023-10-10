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
        
        self.spda = ScaledDotProductAttention()
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """_summary_

        Args:
            query (torch.FloatTensor): shape (batch_size, max_len, d_model)
            key (torch.FloatTensor): shape (batch_size, max_len, d_model)
            value (torch.FloatTensor): shape (batch_size, max_len, d_model)
            mask (Optional[torch.ByteTensor], optional): (batch_size, max_len)

        Returns:
            shape (batch_size, max_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch.
        # x: B x H x L x D_v
        x, self.attn = self.sdpa(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)

if __name__ == "__main__":
    pass