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
    x = torch.rand(2, 10) - 0.5
    activation_layer = nn.Sigmoid()
    y_sig = activation_layer(x)
    print(f"input _x = {x}")
    print(f"output _y_sig = {y_sig}")

    y_tan = nn.Tanh()(x)
    print(f"output _y_tan = {y_tan}")
    
    y_relu = nn.ReLU()(x)
    print(f"output _y_relu = {y_relu}")


if __name__ == "__main__":
    test_activation()
