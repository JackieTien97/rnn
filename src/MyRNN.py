import torch.nn as nn
import torch
import torch.nn.functional as F


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.h = torch.zeros(hidden_size)  # init Hidden state
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Network Parameters
        self.W_xc = nn.Parameter(torch.randn((hidden_size, input_size)), True)
        self.W_hc = nn.Parameter(torch.randn((hidden_size, hidden_size)), True)
        self.b_c = nn.Parameter(torch.zeros(hidden_size), True)

        # Reset gate parameters
        self.W_xr = nn.Parameter(torch.randn_like(self.W_xc), True)
        self.W_hr = nn.Parameter(torch.randn_like(self.W_hc), True)
        self.b_r = nn.Parameter(torch.randn_like(self.b_c), True)

        # Update gate parameters
        self.W_xz = nn.Parameter(torch.randn_like(self.W_xc), True)
        self.W_hz = nn.Parameter(torch.randn_like(self.W_hc), True)
        self.b_z = nn.Parameter(torch.randn_like(self.b_c), True)

    def forward(self, x):
        # Gate updates
        update_gate = F.sigmoid((self.W_xz @ x) + (self.W_hz @ self.h + self.b_z))
        reset_gate = F.sigmoid((self.W_xr @ x) + (self.W_hr @ self.h + self.b_r))

        h_candidate = self.tanh((self.W_xc @ x) + (self.W_hc @ torch.mul(reset_gate, self.h) + self.b_c))

        self.h = torch.mul(1 - update_gate, self.h) + torch.mul(update_gate, h_candidate)
        y_output = self.h @ self.Why + self.by
        return y_output
