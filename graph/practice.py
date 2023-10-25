'''
#################################
# Python API: ML Practice (Transformer Practice - MultiHead Self Attention)
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
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering
import networkx as nx

# from multihead import MultiHeadSelfAttention
# from attention.simpleattention import ScaledDotProductAttention
# from attention.multihead import MultiHeadSelfAttention

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


def test_degree():
    """_summary_

    Returns:
        _type_: _description_
    """
    a = torch.rand(3, 3)
    a[a > 0.5] = 1
    a[a <= 0.5] = 0

    d = calc_degree_matrix(a)
    print('a', a)
    print('d', d)
    return a, d


def test_lapl():
    """_summary_

    Returns:
        _type_: _description_
    """
    a = torch.rand(3, 3)
    a[a > 0.5] = 1
    a[a <= 0.5] = 0
    print('a', a)
    print('lapl', calc_graph_lapl(a))
    return calc_graph_lapl(a)


def test_lalp_norm():
    """_summary_
    """
    a = torch.rand(3, 3)
    print("A: ", a)
    print("L Normalized: ", create_graph_lapl_norm(a))


def test_eigenvalue():
    """_summary_
    """
    g = nx.petersen_graph()
    plt.subplot(121)
    nx.draw(g, with_labels=False, font_weight='bold')
    plt.subplot(122)
    nx.draw_shell(g, nlist=[range(5, 10), range(5)], with_labels=False, font_weight='bold')

    options = {
        'node_color': 'blue',
        'node_size': 100,
        'width': 2,
    }

    plt.figure()
    plt.subplot(221)
    nx.draw_random(g, **options)
    plt.subplot(222)
    nx.draw_circular(g, **options)
    plt.subplot(223)
    nx.draw_spectral(g, **options)
    plt.subplot(224)
    nx.draw_shell(g, nlist=[range(5, 10), range(5)], **options)
    print(' *********** DONE ************ ')


def segmentation_laplacian():
    
    
    
if __name__ == "__main__":
    # test_lapl()
    # test_lalp_norm()
    test_eigenvalue()
