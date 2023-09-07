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
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
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


if __name__ == "__main__":
    attention_model = Attention(y_dim=10, h_dim=20)
    y = torch.rand(1, 10)
    h = torch.rand(1, 20)
    print(attention_model(y=y, h=h))
    # train(model, training_data)
