'''
#################################
# Python API: ML Practice (CNN)
#################################
'''

#########################################################
# import libraries
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd + '/../cifar10')

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils
from torchvision import transforms
import torch
import torch.nn as nn

from Cifar10Dataloader import CIFAR10


#########################################################
# General Parameters
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batch_size = 4

#########################################################
# Function definition

def show_image(img):
    """_summary_

    Args:
        img (_type_): _description_
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    #convert the images to tensor and normalized them
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = CIFAR10(root='../cifar10',  transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)
    return trainloader

class CNN(nn.Module):
    def __init__(self):
        """_summary_
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, training_data):
    """_summary_
    Args:
        model (_type_): _description_
        training_data (_type_): _description_
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(training_data, 0):
            # get the inputs; cifar10 is a list of [inputs, labels]
            inputs, targets = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def evaluate(model):
    """_summary_

    Args:
        model (_type_): _description_
    """
    dataiter = iter(load_data())
    images, labels = dataiter.next()

    # print images
    show_image(utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    outputs = model(images)
    
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


def buildcnn():


if __name__ == "__main__":
    buildcnn()