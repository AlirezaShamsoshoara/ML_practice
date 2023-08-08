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


"""


def auto_encoder():
    """_summary_
    """
    pass


if __name__ == "__main__":
    auto_encoder()
