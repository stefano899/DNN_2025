import torch
from torch import nn


class SetA1(nn.Module):
    """
    Class SetA1: Defines the first architecture.

    Architecture Details:
        - One convolutional layer with 5 output channels, kernel size 3x3, padding=1, and stride=1.
        - One max pooling layer (non-trainable).
        - One fully connected layer with an input size of 980 and an output size of 10.
        - Activation function: ReLU.
        - Total number of trainable parameters: 5×14×14 × 10 + 3×3×5 = 9.845
    """

    def __init__(self, classes=10):
        super(SetA1, self).__init__()

        torch.manual_seed(0)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 14 * 14,
                             classes)  # first number is to decoding the 3d tensor vector into a 1D dimensional vector, 100 is the number of output neurons of fc1. I chosed 100 because is a good tradeoff between speed and leaning capacity
        self.relu = nn.ReLU()  # Activation Function
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))

        x = self.flatten(x)

        x = self.fc1(x)

        return x
