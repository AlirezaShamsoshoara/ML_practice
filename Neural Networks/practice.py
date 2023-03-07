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
    classifier = nn.Linear(10,3)
    loss = nn.MSELoss()

    input_vector = torch.randn(10)
    target = torch.tensor([0, 0, 1])
    pred = classifier(input_vector)
    loss = loss(pred, target)
    print("Prediction: " ,pred)
    print("loss: " , loss)


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


if __name__ == "__main__":
    linear_classifier()
    train()
