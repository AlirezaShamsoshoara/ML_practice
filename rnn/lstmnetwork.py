'''
#################################
# Python API: ML Practice (LSTM Network)
#################################
'''

#########################################################
# import libraries
import random
import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

import torch.optim as optim

#########################################################
# General Parameters
#set seed to be able to replicate the resutls
seed = 172
random.seed(seed)
torch.manual_seed(seed)

#########################################################
# Function definition

"""_summary_
We connect LSTM cells in time by connecting both the context vector and the hidden state vector
from the previous timestep (the so-called unrolling).  Recurrent models are really flexible in the mapping
from input to output sequences. You just have to modify the input to hidden states and the hidden to output
states based on the problem. By definition, LSTMs can process arbitrary input timesteps. The output can be
tuned by designing which outputs of the last hidden-to-hidden layer are used to compute the desired output.
"""

def generate_sin_wave_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    T = 20
    L = 1000
    N = 200

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    return data


class Sequence(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(Sequence, self).__init__()

        self.rnn1 = nn.LSTMCell(1, 51)
        self.rnn2 = nn.LSTMCell(51, 51)

        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        """_summary_

        Args:
            input (_type_): _description_
            future (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            h_t, c_t = self.rnn1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)
            outputs += [output]

        # if we should predict the future
        for i in range(future):

            h_t, c_t = self.rnn1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def train():
    """_summary_

    Returns:
        _type_: _description_
    """
    # load data and make training set
    data = generate_sin_wave_data()
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    seq = Sequence()

    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)

    # begin to train
    for i in range(1):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()

        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.show()


if __name__ == "__main__":
    generate_sin_wave_data()
    train()
