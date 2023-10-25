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


if __name__ == "__main__":
    pass
