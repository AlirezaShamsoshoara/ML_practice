'''
#################################
# Python API: ML Practice (Training Neural Network)
#################################
'''

#########################################################
# import libraries
import torch
import torch.nn as nn

#########################################################
# General Parameters

#########################################################
# Function definition

def test_activation():
    """_summary_
    """
    activation_layer = nn.Sigmoid()
    x = torch.rand(2, 10)
    y_sig = activation_layer(x)
    print(f"input _x = {x}")
    print(f"output _y_sig = {y_sig}")

    y_tan = nn.Tanh()(x)
    print(f"output _y_tan = {y_tan}")


if __name__ == "__main__":
    test_activation()
