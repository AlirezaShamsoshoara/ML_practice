'''
#################################
# Python API: ML Practice (CNN)
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


def test_cnn():
    input_img = torch.rand(1,3,7,7)
    layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1)
    out = layer(input_img)
    print(out.shape)


if __name__ == "__main__":
    test_cnn()
