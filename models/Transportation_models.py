import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvMaxPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.01):
        super(ConvMaxPoolBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=kernel_size//2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha, inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)
        return x


class TransportationCNN(nn.Module):
    def __init__(self, n_classes):
        super(TransportationCNN, self).__init__()
        # Network from "A Convolutional Neural Network for Transportation Mode Detection Based on Smartphone Platform"
        # https://ieeexplore.ieee.org/abstract/document/8108764
        alpha = 0.01

        # (1 x 512) --> (32 x 256)
        self.conv1 = ConvMaxPoolBlock(in_channels=1, out_channels=32, kernel_size=15, alpha=alpha)
        # next two convs have kernel size 10
        # (32 x 256) --> (64 x 128)
        self.conv2 = ConvMaxPoolBlock(in_channels=32, out_channels=64, kernel_size=10, alpha=alpha)
        # (64 x 128) --> (64 x 64)
        self.conv3 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=10, alpha=alpha)
        # last 4 convs have kernel size 5
        # (64 x 64) --> (64 x 32)
        self.conv4 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5, alpha=alpha)
        # (64 x 32) --> (64 x 16)
        self.conv5 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5, alpha=alpha)
        # (64 x 16) --> (64 x 8)
        self.conv6 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5, alpha=alpha)
        # The paper states there is a first convolution with kernel_size=15 followed by with maxpool,
        # two more with kernel_size 10, then 4 more with kernel_size 5. But the last conv would create a 64 x 4 tensor,
        # so we dont do it. TODO Maybe ask authors
        # (64 x 8) --> (64 x 4)
        # self.conv7 = ConvMaxPoolBlock(in_channels=64, out_channels=64, kernel_size=5, alpha=alpha)

        # Now into a fully connected layer with 200 neurons
        self.flatten = Flatten()
        self.hidden = nn.Linear(in_features=64*8, out_features=200)
        self.relu = nn.LeakyReLU(negative_slope=alpha, inplace=True)
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
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    import numpy as np
    from torchsummary import summary
    model = TransportationCNN(n_classes=7)
    summary(model, input_size=(1, 512), device="cpu")
    # Shape: Batch x Channels x Feature-length
    inp = torch.rand(1, 1, 512)
    y_pred = model(inp)
