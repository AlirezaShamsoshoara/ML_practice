'''
#################################
# Python API: ML Practice (Autoencoder Practice)
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
For the decoder, we will use a simple network with 2 linear layers. This network will parameterize the variational posterior 
. We also need another neural network (the Encoder), which will parameterize the likelihood 
, again with 2 linear layers.
"""

class VAE(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(VAE, self).__init__()
        self.features = 16
        
        #encoder
        self.enc1 = nn.Linear(in_features=3072, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=self.features * 2)
        
        # decoder
        self.dec1 = nn.Linear(in_features=self.features, out_features=128)
        self.dec2 = nn.Linear(in_features=128, out_features=3072)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # generate sample of the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.features)
        mu = x[:, 0, :]
        log_var = x[:, 1, :]
        z = self.reparameterize(mu, log_var)

        x = F.relu(self.dec1(z))
        reconst = torch.sigmoid(self.dec2(x))
        return reconst, mu, log_var


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, training_data):
    """_summary_

    Args:
        model (_type_): _description_
        training_data (_type_): _description_
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss(reduction='sum')
    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(training_data, 0):
            inputs, _ = data
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(inputs)
            bce_loss = criterion(reconstruction, inputs)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == "__main__":
    model = VAE()
    train(model, training_data)
