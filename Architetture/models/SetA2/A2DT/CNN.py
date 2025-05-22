from torch import nn
import torch
from torch.nn import init


class A2DT(nn.Module):
    def __init__(self, classes):
        super(A2DT, self).__init__()
        self.name = "A2DT"
        self.set = "A2"
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(10 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, len(classes))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()  # Activation Function
        self.flatten = nn.Flatten()
        self.initialize_default_weights()

    def initialize_default_weights(self):
        with torch.no_grad():
            init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
            init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
            init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
