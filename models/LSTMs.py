import torch
from torch import nn


class MySmallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MySmallLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.lin = nn.Linear(in_features=self.hidden_size, out_features=output_size)

    def forward(self, x):
        out, state = self.lstm(x)
        out = self.lin(out[:, -1, :])
        out = torch.sigmoid(out)
        return out


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size // 2, num_layers=1,
                             batch_first=True)
        self.lin1 = nn.Linear(in_features=self.hidden_size // 2, out_features=self.hidden_size // 4)
        self.lin2 = nn.Linear(in_features=self.hidden_size // 4, out_features=self.output_size)

    def forward(self, x):
        out, state = self.lstm1(x)
        out, state = self.lstm2(out)
        # Only take the last state of the second LSTM
        out = self.lin1(out[:, -1, :])
        out = torch.sigmoid(out)
        out = self.lin2(out)
        out = torch.sigmoid(out)
        return out
