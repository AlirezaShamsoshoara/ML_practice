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
        self.linear_gate_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        # 2. DEFINE CELL MEMORY COMPONENTS
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_gate = nn.Tanh()


        # out gate components
        # 3. DEFINE OUT GATE COMPONENTS
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_hidden_out = nn.Sigmoid()


        # final output
        # 4. DEFINE OUTPUT
        self.activation_final = nn.Tanh()

    def forget(self, x, h):
        x_temp = self.linear_gate_w1(x)
        h_temp = self.linear_gate_r1(h)
        f = self.sigmoid_forget(x_temp + h_temp)
        return f

    def input_gate(self, x, h):
        # input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)
        return i

    def cell_memory_gate(self, i, f, x, g, c_prev):
        x = self.linear_gate_w3(x)
        h = self.linear_gate_r3(h)

        # new information part that will be injected in the new context
        k = self.activation_gate(x + h)
        g = k * i
        # forget old context/cell info
        c = f * c_prev
        # learn new context/cell info
        c_next = g + c
        return c_next

    def out_gate(self, x, h):
        x = self.linear_gate_w4(x)
        h = self.linear_gate_r4(h)
        return self.sigmoid_hidden_out(x + h)

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
    """_summary_
    """
    layer = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
    return layer


if __name__ == "__main__":
    layer = lstm_test()
    lst_obj = MyLstmCell()
    print(lst_obj)
