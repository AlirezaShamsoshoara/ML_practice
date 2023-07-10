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

def batch_norm(x, gamma, beta):
    """_summary_

    Args:
        x (_type_): _description_
        gamma (_type_): _description_
        beta (_type_): _description_

    Returns:
        _type_: _description_
    """
    # N, C, H, W
    N, C, H, W = list(x.size())
    mean = torch.mean(x, dim=(0, 2, 3))
    variance = torch.mean((x - mean.reshape((1, C, 1, 1))) ** 2, dim=(0, 2, 3))
    x_hat = (x - mean.reshape((1, C, 1, 1))) * 1.0 / torch.sqrt(variance.reshape((1, C, 1, 1)))
    norm_x = gamma.reshape((1, C, 1, 1)) * x_hat + beta.reshape((1, C, 1, 1))
    return norm_x


def drop_out():
    """_summary_
    """
    input_data = torch.rand(1, 8)
    layer_dp = nn.Dropout(0.5)
    out1 = layer_dp(input_data)
    out2 = layer_dp(input_data)
    print(f"out1 = {out1} \n out2 = {out2}")


if __name__ == "__main__":
    # batch_norm(torch.rand(8, 3, 40, 80),
    #            gamma=torch.ones(8, 3, 40, 80),
    #            beta=torch.ones(8, 3, 40, 80))
    drop_out()
