from torch import nn
import torch


class A1HT(nn.Module):
    def __init__(self, classes):
        super(A1HT, self).__init__()
        self.name = "A1HT"
        self.set = "A1"
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Undersampling
        self.fc1 = nn.Linear(5 * 14 * 14,
                             200)  # Fully connected layer
        self.fc2 = nn.Linear(200,
                             len(classes))
        self.relu = nn.ReLU()  # Activation Function
        self.flatten = nn.Flatten()
        self.set_initial_kernels()
        self.initialize_default_weights()

    def set_initial_kernels(self):
        # Definito by hand i primi 5 kernels
        kernel1 = torch.tensor([[0, 0, 0],
                                [1, 1, 1],
                                [0, 0, 0]], dtype=torch.float32)

        kernel2 = torch.tensor([[0, 1, 0],
                                [0, 1, 0],
                                [0, 1, 0]], dtype=torch.float32)

        kernel3 = torch.tensor([[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]], dtype=torch.float32)

        kernel4 = torch.tensor([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]], dtype=torch.float32)

        kernel5 = torch.tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=torch.float32)
        kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]  # list of
        with torch.no_grad():
            for k, kernel in enumerate(kernels):
                self.conv1.weight[k, 0] = kernel

    def initialize_default_weights(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        return x

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
