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

def create_adj(size):
    a = torch.rand(size, size)
    a[a > 0.5] = 1
    a[a <= 0.5] = 0
    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i ==j:
                a[i, j] = 0
    return a


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


def create_graph_lapl_norm(a):
    """_summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """
    size = a.shape[-1]
    d_norm = calc_degree_matrix_norm(a)
    l_norm = torch.ones(size) - (d_norm @ a @ d_norm)
    return l_norm


def find_eigmax(L):
    """_summary_

    Args:
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        # e1, _ = torch.eig(L, eigenvectors=False)
        e1, _ = torch.linalg.eig(L, )
        eig_magnitudes = torch.abs(e1)
        max_magnitude = eig_magnitudes.max().item()
        return max_magnitude


def chebyshev_lapl(X, lapl, thetas, order):
    """_summary_

    Args:
        X (_type_): _description_
        lapl (_type_): _description_
        thetas (_type_): _description_
        order (_type_): _description_

    Returns:
        _type_: _description_
    """
    list_powers = []
    nodes = lapl.shape[0]
    t0 = X.float()
    eigmax = find_eigmax(lapl)
    l_rescaled = (2 * lapl / eigmax) - torch.eye(nodes)
    
    y = t0 * thetas[0]
    list_powers.append(y)
    t1 = torch.matmul(l_rescaled, t0)
    list_powers.append(t1 * thetas[1])

    # Computation of: T_k = 2*L_rescaled*T_k-1  -  T_k-2
    for k in range(2, order):
        t2 = 2 * torch.matmul(l_rescaled, t1) - t0
        list_powers.append((t2 * thetas[k]))
        t0, t1, = t1, t2
    y_out = torch.stack(list_powers, dim=-1)
    # the powers may be summed or concatenated. i 
    # use concatenation here
    y_out = y_out.view(nodes, -1) # -1 = order * features_of_signal
    return y_out


def test_gcn():
    features = 3
    out_features = 50
    a = create_adj(10)
    L = create_graph_lapl_norm(a)
    x = torch.rand(10, features)
    power_order = 4 # p-hops
    thetas = nn.Parameter(torch.rand(4))
    out = chebyshev_lapl(x,L,thetas,power_order)
    print('cheb approx out powers concatenated:', out.shape)
    #because we used concatenation  of the powers
    #the out features will be power_order * features
    linear = nn.Linear(4*3, out_features)
    layer_out = linear(out)
    print('Layers output:', layer_out.shape)
    
if __name__ == "__main__":
    test_gcn()
