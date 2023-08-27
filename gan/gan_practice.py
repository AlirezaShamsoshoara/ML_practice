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

        self.out = nn.Sequential(
            nn.Linear(in_features=256, out_features=n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        return self.out(x)


class GenNet(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, ):
        super(GenNet, self).__init__()
        n_features = 100
        n_out = 3072

        self.hidden_0 = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=256),
            nn.LeakyReLU(0.2)
        )

        self.hidden_1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2)
        )

        self.hidden_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(in_features=1024, out_features=n_out),
            nn.Tanh()
        )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        return self.out(x)


def train_disc(disc, opt, real_data,  fake_data, loss):
    """_summary_

    Args:
        disc (_type_): _description_
        opt (_type_): _description_
        real_data (_type_): _description_
        fake_data (_type_): _description_
        loss (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = real_data.size(0)
    opt.zero_grad()

    pred_real = disc(real_data)
    loss_real = loss(pred_real, ones_target(n))
    loss_real.backward()

    pred_fake = disc(fake_data)
    loss_fake = loss(pred_fake, zero_targets(n))
    loss_fake.backward()

    opt.step()

    return loss_real + loss_fake, pred_real, pred_fake


def train_gen(disc, opt, fake_data, loss):
    """_summary_

    Args:
        disc (_type_): _description_
        opt (_type_): _description_
        fake_data (_type_): _description_
        loss (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = fake_data.size(0)
    opt.zero_grad()

    pred = disc(fake_data)
    loss_gen = loss(pred, ones_target(n))
    loss_gen.backward()

    opt.step()
    return loss_gen


def noise(size):
    n = torch.randnn(size, 100)
    return n


def ones_target(size):
    data = torch.ones(size, 1)
    return data


def zero_targets(size):
    data = torch.zeros(size, 1)
    return data


def image_to_vectors(image):
    return image.view(image.size(0), -1)


def load_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    train_set = datasets.CIFAR10(root='../cifar10', train=True, 
                                 download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=False, 
                                               num_workers=1)

    return train_loader


def gan_practice():
    """_summary_
    """
    disc = DiscNet()
    gen = GenNet()
    
    loss_d = nn.BCELoss()
    loss_g = nn.BCELoss()
    
    opt_d = optim.Adam(disc.parameters(), lr=0.0002)
    opt_g = optim.Adam(gen.parameters(), lr=0.0002)
    
    data_loader = load_data()
    
    for epoch in range(1):
        for n_batch, data in enumerate(data_loader):
            
            (real_batch, labels) = data
            n = real_batch.size(0)
            
            real_data = image_to_vectors(real_batch)
            latent_space_data = noise(n)
            fake_data = gen(latent_space_data).detach()
            
            # loss_real + loss_fake, pred_real, pred_fake
            loss_d, pred_real, pred_fake = train_disc(disc, opt=opt_d, real_data=real_data,
                                                      fake_data=fake_data, loss=loss_d)
            
            latent_space_data = noise(n)
            fake_data = gen(latent_space_data)
            loss_g = train_gen(disc=disc, opt=opt_g, fake_data=fake_data, loss=loss_g)
            print(f'Loss _ D:{loss_d}, Loss _ G:{loss_g}')

if __name__ == "__main__":
    gan_practice()
    # train(model, training_data)
