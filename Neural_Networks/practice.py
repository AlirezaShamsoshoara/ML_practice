'''
#################################
# Python API: ML Practice (Neural Network)
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


def linear_classifier():
    """_summary_
    """
    classifier = nn.Linear(10, 3)
    loss = nn.MSELoss()

    input_vector = torch.randn(10)
    target = torch.tensor([0, 0, 1])
    pred = classifier(input_vector)
    loss = loss(pred, target)
    print("Prediction: ", pred)
    print("loss: ", loss)


def train():
    """_summary_
    """
    model = nn.Linear(4, 2)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        inputs = torch.Tensor([0.8, 0.4, 0.4, 0.2])
        labels = torch.Tensor([1, 0])

        # Clear gradient buffers because we don't want any gradient
        # from previous epoch to carry forward
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # print(f"Loss = {loss}")

        loss.backward()
        optimizer.step()

        print(f'epoch {epoch}, loss {loss.item()}')


def neuron(input_val):
    """_summary_

    Args:
        input (_type_): _description_

    Returns:
        _type_: _description_
    """
    # WRITE YOUR CODE HERE
    w = torch.tensor([0.5, 0.5, 0.5])
    b = torch.tensor([0.5])
    return torch.add(torch.matmul(input_val, w), b)


class Model(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(5, 10)
        self.linear2 = nn.Linear(10, 2)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        h = torch.sigmoid(self.linear1(x))
        o = torch.sigmoid(self.linear2(h))
        return o, h


def test():
    """_summary_
    """
    seed = 172
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([[1, 2], [3, 4]])
    model = Model()
    x = torch.randn((1, 5))
    o, h = model(x)


def fnn(input):
    """_summary_

    Args:
        input (_type_): _description_
    """
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    return model(input)


if __name__ == "__main__":
    linear_classifier()
    # train()
    print(neuron(torch.tensor([1.0, 1.0, 1.0])))
    print(test())
