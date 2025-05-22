from torch import nn
import torch
from torch.nn import init


class A1DT(nn.Module):
    def __init__(self, classes):
        super(A1DT, self).__init__()
        self.name = "DT"
        self.set = "A1"
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 14 * 14,
                             200)  # first number is to decoding the 3d tensor vector into a 1D dimensional vector, 100 is the number of output neurons of fc1. I chosed 100 because is a good tradeoff between speed and leaning capacity
        self.fc2 = nn.Linear(200,
                             len(classes))  # Final fully connected layer. This is the layer that makes the preictions.
        self.relu = nn.ReLU()  # Activation Function
        self.flatten = nn.Flatten()
        self.initialize_default_weights()


    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        return x

    def initialize_default_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set