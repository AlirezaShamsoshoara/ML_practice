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


class DiscNet(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, ):
        super(DiscNet, self).__init__()
        n_features = 3072
        n_out = 1

        self.hidden_0 = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=1024, ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.out == nn.Sequential(
            nn.Linear(in_features=256, out_features=n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        return self.out(x)


class GenNet(torch.nn.Module):
    def __init__(self,):
        super(GenNet, self).__init__()
        n_features = 100
        n_out = 3072
        

def gan_practice():
    pass


if __name__ == "__main__":
    gan_practice()
    # train(model, training_data)
