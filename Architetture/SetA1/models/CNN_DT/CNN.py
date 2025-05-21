from torch import nn
import torch


class Net(nn.Module):
    def __init__(self, classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 14 * 14, 100)  # first number is to decoding the 3d tensor vector into a 1D dimensional vector, 100 is the number of output neurons of fc1. I chosed 100 because is a good tradeoff between speed and leaning capacity
        self.fc2 = nn.Linear(100, len(classes))  # Final fully connected layer. This is the layer that makes the preictions.
        self.relu = nn.ReLU()  # Activation Function
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))

        x = self.flatten(x)  # x becomes a 2D vector

        x = self.relu(self.fc1(x))  # actv. function applied on the first fully connected layer of output

        x = self.fc2(x)  # Prediction
        return x


