'''
#################################
# Python API: ML Practice (RNN Simple)
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

"""_summary_
As we already know, convolutional layers are specialized for processing grid-structured data (i.e., images).
On the contrary, recurrent layers are designed for processing sequences.
To distinguish Recurrent NN (RNNs) from fully-connected layers, we call the non-recurrent networks feedforward NN.
The smallest computational unit in recurrent networks is the cell. The cell is the basic unit in recurrent networks.
Recurrent cells are NN for processing sequential data. They are usually small.
Recurrent models help us deal with time-varying signals, so always have the notion of time and timesteps in the back of your mind.
One can create a minimal recurrent unit by connecting the current timesteps' output to the input of the next timestep!
We choose to model the time dimension with RNNs because we want to learn temporal and often long-term dependencies.
The majority of common recurrent cells can also process sequences of variable length. This is really important for many applications
, including videos that contain a different number of images.
One can view the RNN cell as a common NN with shared weights for the multiple timesteps. With this modification,
the weights of the cell now have access to the previous states of the sequence.
That’s why input unrolling is the only way we can make backpropagation work!
In essence, backpropagation requires a separate layer for each time step with the same weights for all layers (input unrolling)!
 we need a separate layer for each time step with the same weights for all layers.
Sequence unrolling and backpropagation through time come hand in hand. This is also why RNN layers are slow at training.

 
"""
dataloader = None

def rnn_roling_test():
    accumulate_gradient_steps = 2
    for counter, data in enumerate(dataloader):
        inputs, targets = data
        predictions = model(inputs)
        loss = criterion(predictions, targets)/accumulate_gradient_steps
        loss.backward()
        
        if counter % accumulate_gradient_steps ==0:
            optimizer.step()
            optimizer.zero_grad() 

if __name__ == "__main__":
    rnn_roling_test()