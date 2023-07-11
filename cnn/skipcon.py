'''
#################################
# Python API: ML Practice (Skip Connection)
#################################
'''

#########################################################
# import libraries
import torch
import torch.nn as nn


#########################################################
# General Parameters
SEED = 172
torch.manual_seed(seed=SEED)

#########################################################
# Function definition

# Skip connection via add (Resnet) or Concatanation (DenseNet)

class SkipConnection(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(SkipConnection).__init__(*args, **kwargs)
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=6,
                                      kernel_size=2, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(in_channels=6, out_channels=3,
                                     kernel_size=2, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input_x):
        """_summary_

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.relu(self.conv_layer1(input_x))
        x = self.relu2(self.conv_layer2(x))
        # output = torch.cat([input, x])
        output = input_x + x
        return output


if __name__ == "__main__":
    model = SkipConnection()