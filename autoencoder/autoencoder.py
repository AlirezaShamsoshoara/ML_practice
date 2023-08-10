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
Discriminative models learn the probability of a label based on a data point 
In mathematical terms, this is denoted as p(y∣x). In order to categorize a data
point into a class, we need to learn a mapping between the data and the classes.
This mapping can be described as a probability distribution. Each label will “compete”
with the other ones for probability density over a specific data point.

Generative models, on the other hand, learn a probability distribution over the data points without external labels.
Mathematically, this is formulated as p(x). In this case, we have the data themselves “compete” for probability density.


Conditional Generative models are another category of models that try to learn the probability distribution of the data 
conditioned on the labels y. As you can probably tell, this is denoted as p(x∣y).
Here, we again have the data “compete” for density, but for each possible label.

Note that in most cases, generative models are unsupervised, meaning that they are trained using unlabeled data.

Latent variable models aim to model the probability distribution with latent variables.

Latent variables are a transformation of the data points into a continuous lower-dimensional space.


Autoencoders are simple neural networks such that their output is their input.
The first part of the network is what we refer to as the encoder. It receives the input and encodes it in a latent
space of a lower dimension (the latent variables z).
For now, you can think of the latent space as a continuous low-dimensional space.
The second part (the decoder) takes that vector and decodes it in order to produce the original input.
The latent vector z in the middle is what we want, as it is a compressed representation of the input.
The applications of the latent vector z are plentiful.
""" 

class Autoencoder(nn.Module):
    def __init__(self):
        """
        In the first part of the network, the size of the input is gradually decreasing,
        resulting in a compact latent representation. In the second part, ConvTrasnposed2d layers
        are increasing the size with the goal to output the original size on the final layer.
        """
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


def auto_encoder():
    """_summary_
    """
    pass


if __name__ == "__main__":
    auto_encoder()
