import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvMaxPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation_func="elu", alpha=0.01):
        super(ConvMaxPoolBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=kernel_size//2)
        if activation_func == "relu":
            self.activation_func = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        elif activation_func == "elu":
            self.activation_func = nn.ELU(alpha=alpha, inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation_func(x)
        x = self.maxpool(x)
        return x


class TransportationCNN(nn.Module):
    def __init__(self, in_channels, n_classes, activation_function="elu", alpha=0.01):
        super(TransportationCNN, self).__init__()
        # Network from "A Convolutional Neural Network for Transportation Mode Detection Based on Smartphone Platform"
        # https://ieeexplore.ieee.org/abstract/document/8108764

        # (input_channels x 512) --> (32 x 256)
        self.conv1 = ConvMaxPoolBlock(in_channels=in_channels, out_channels=32, kernel_size=15,
                                      activation_func=activation_function, alpha=alpha)
        # next two convs have kernel size 10
        # (32 x 256) --> (64 x 128)
        self.conv2 = ConvMaxPoolBlock(in_channels=32, out_channels=64, kernel_size=10,
                                      activation_func=activation_function, alpha=alpha)
        # (64 x 128) --> (64 x 64)
        self.conv3 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=10,
                                      activation_func=activation_function, alpha=alpha)
        # last 4 convs have kernel size 5
        # (64 x 64) --> (64 x 32)
        self.conv4 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5,
                                      activation_func=activation_function, alpha=alpha)
        # (64 x 32) --> (64 x 16)
        self.conv5 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5,
                                      activation_func=activation_function, alpha=alpha)
        # (64 x 16) --> (64 x 8)
        self.conv6 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5,
                                      activation_func=activation_function, alpha=alpha)
        # The paper states there is a first convolution with kernel_size=15 followed by with maxpool,
        # two more with kernel_size 10, then 4 more with kernel_size 5. But the last conv would create a 64 x 4 tensor,
        # so we dont do it. TODO Maybe ask authors
        # (64 x 8) --> (64 x 4)
        # self.conv7 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5, alpha=alpha)

        # Now into a fully connected layer with 200 neurons
        self.flatten = Flatten()
        self.hidden = nn.Linear(in_features=64*8, out_features=200)
        if activation_function == "relu":
            self.activation_func = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        elif activation_function == "elu":
            self.activation_func = nn.ELU(alpha=alpha, inplace=True)
        # to the output layer
        self.output = nn.Linear(in_features=200, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # x = self.conv7(x)
        # flatten the output
        x = self.flatten(x)
        x = self.activation_func(self.hidden(x))
        x = self.output(x)
        return x


class TransportationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=False):
        super(TransportationLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Without this using bi-directional memory requirements is increased significantly
        factor = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size // factor,
                             bidirectional=bidirectional,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size // 2 // factor,
                             bidirectional=bidirectional,
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
        return out


if __name__ == "__main__":
    import numpy as np
    from torchsummary import summary
    model = TransportationCNN(n_classes=7)
    summary(model, input_size=(1, 512), device="cpu")
    # Shape: Batch x Channels x Feature-length
    inp = torch.rand(1, 1, 512)
    y_pred = model(inp)
