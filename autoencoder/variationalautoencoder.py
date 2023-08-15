'''
#################################
# Python API: ML Practice (Basic Autoencoder)
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
# %matplotlib inline

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
In simple terms, a variational autoencoder is a probabilistic version of autoencoders.
Because we want to be able to sample from the latent vector (z) space to generate new data,
which is not possible with vanilla autoencoders.
"""

def variational_auto_encoder():
    """_summary_
    """
    pass


def elbo(reconstructed, input, mu, logvar):
    """
    Args:
        `reconstructed`: The reconstructed input of size [B, C, W, H],
        `input`: The oriinal input of size [B, C, W, H],
        `mu`: The mean of the Gaussian of size [N], where N is the latent dimension
        `logvar`: The log of the variance of the Gaussian of size [N], where N is the latent dimension

    Returns:
        a scalar
    """
    bce_loss = nn.BCELoss()
    rec_loss = bce_loss(reconstructed, input)
    kl_loss = -0.5 * (torch.sum(1+logvar - mu.pow(2) - logvar.exp()))
    return rec_loss + kl_loss

if __name__ == "__main__":
    variational_auto_encoder()
