'''
#################################
# Python API: ML Practice (Training Neural Network)
#################################
'''

#########################################################
# import libraries
import torch
import torch.nn as nn
import os
import sys
cwd = os.getcwd()
#add CIFAR10 data in the environment
sys.path.append(cwd + '/../cifar10') 

#Numpy is linear algebra lbrary
import numpy as np
# Matplotlib is a visualizations library 
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import utils
from torchvision import transforms
#CIFAR10 is a custom Dataloader that loads a subset ofthe data from a local folder
from Cifar10Dataloader import CIFAR10

#########################################################
# General Parameters

#########################################################
# Function definition

def test_activation():
    """_summary_
    """
    x = torch.rand(2, 10) - 0.5
    activation_layer = nn.Sigmoid()
    y_sig = activation_layer(x)
    print(f"input _x = {x}")
    print(f"output _y_sig = {y_sig}")

    y_tan = nn.Tanh()(x)
    print(f"output _y_tan = {y_tan}")

    y_relu = nn.ReLU()(x)
    print(f"output _y_relu = {y_relu}")

    y_leakyrelu = nn.LeakyReLU()(x)
    print(f"output _y_leakyrelu = {y_leakyrelu}")

    y_prelu = nn.PReLU()(x)
    print(f"input _x = {x}")
    print(f"output _y_prelu = {y_prelu}")

    y_soft = nn.Softmax(dim=-1)(x)
    print(f"input _x = {x}")
    print(f"output _y_soft = {y_soft}")

    y_m_sigmoid = m_sigmoid(x=x)
    y_m_tan = m_tanh(x=x)
    y_m_relu = m_relu(x=x)
    y_m_softmax = m_softmax(x=x)


def m_sigmoid(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 / (1 + torch.exp(-x))


def m_tanh(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def m_relu(x):
    """""_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """""
    # zeros = torch.zeros_like(x)
    return torch.maximum(x, torch.tensor(0))
    # return x * (x > 0)


def m_softmax(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (torch.exp(x)) / (torch.sum(torch.exp(x)))


model = nn.Sequential(nn.Linear(3072, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10)
                          )
batch_size=4
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


def show_image(img):
    """_summary_

    Args:
        img (_type_): _description_
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

classes = ('plane', 'car', 'bird', 'cat',
       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(load_data())
images, labels = dataiter.next()

# show images
show_image(utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


## 1. DEFINE MODEL HERE


def train():
    """_summary_
    """
    training_data = load_data()

    # 2. LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0

    for epoch in range(10):
        for i, data in enumerate(training_data, 0):

            inputs, labels = data
            #reshape images so they can be fed to a nn.Linear()
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()

            ##3. RUN BACKPROP
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Training finished')


def evaluate():
    """_summary_
    """
    dataiter = iter(load_data())
    images, labels = dataiter.next()

    # print images
    show_image(utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    images = images.view(images.size(0), -1)
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

if __name__ == "__main__":
    test_activation()
