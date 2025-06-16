from torch import nn
import torch


class SetA2(nn.Module):
    """
    Class SetA2: Defines the second type of architecture.

    Architecture Overview:
        - Two convolutional layers:
            - The first has 5 output channels, kernel size 3x3, padding=1, and stride=1.
            - The second has 10 output channels, 5 input channels, kernel size 3x3, padding=0, and stride=1.
        - Two max pooling layers (non-trainable).
        - One fully connected layer with an input size of 360 and an output size of 10.
        - Activation function: ReLU.
        - Total number of trainable parameters: 5*6*6 * 10 + 3*3*5 + 10*3*3 = 1800 + 45 + 90 = 1.935
    """

    def __init__(self, classes=10):
        super(SetA2, self).__init__()
        torch.manual_seed(0)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(10 * 6 * 6, classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

