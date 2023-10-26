'''
#################################
# Python API: ML Practice (Graph Convolutional Network)
#################################
'''

#########################################################
# import libraries
# from locale import nl_langinfo
import random
import numpy as np
import torch
import math
import copy
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections.abc import Sequence
from typing import Tuple, Optional, Union
# %matplotlib inline
# from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from copy import deepcopy
from scipy import misc
from skimage.transform import resize
from numpy import linalg as LA
from scipy.sparse import csgraph
import scipy.sparse as sparse
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering
import networkx as nx

#########################################################
# General Parameters
#set seed to be able to replicate the resutls
seed = 172
random.seed(seed)
torch.manual_seed(seed)

#########################################################
# Function definition

def calc_degree_matrix(a):
    """_summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.diag(a.sum(dim=-1))


def calc_graph_lapl(a):
    """_summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    return calc_degree_matrix(a) - a


def calc_degree_matrix_norm(a):
    """_summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.diag(torch.pow(a.sum(dim=-1), -0.5))


def find_eigmax(L):
    """_summary_

    Args:
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        e1, _ = torch.eig(L, eigenvectors=False)
        return torch.max(e1[:, 0]).item()


def chebyshev_lapl(X, lapl, thetas, order):
    list_powers = []
    nodes = lapl.shape[0]
    t0 = x.float()
    eigmax = 
    


if __name__ == "__main__":
    pass
