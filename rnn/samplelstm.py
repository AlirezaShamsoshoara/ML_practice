'''
#################################
# Python API: ML Practice (LSTM sample)
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

class MyLstmCell(torch.nn.Module):
    def __init__(self, input_length=10, hidden_length=20):
        super(MyLstmCell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components
        # 1. DEFINE FORGET GATE COMPONENTS


        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        # 2. DEFINE CELL MEMORY COMPONENTS


        # out gate components
        # 3. DEFINE OUT GATE COMPONENTS


        # final output
        # 4. DEFINE OUTPUT

    def forget(self, x, h):
        pass

    def input_gate(self, x, h):
        # input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)
        return i

    def cell_memory_gate(self, i, f, x, g, c_prev):
        pass

    def out_gate(self, x, h):
        pass

    def forward(self, x, tuple_in ):
        (h, c_prev) = tuple_in
        # Equation 1. input gate
        i = self.input_gate(x, h)

        # Equation 2. forget gate
        f = self.forget(x, h)

        # Equation 3. updating the cell memory
        c_next = self.cell_memory_gate(i, f, x, h,c_prev)

        # Equation 4. calculate the main output gate
        o = self.out_gate(x, h)

        # Equation 5. produce next hidden output
        h_next = o * self.activation_final(c_next)

        return h_next, c_next

def lstm_test():
    layer = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
    

if __name__ == "__main__":
    lstm_test()
