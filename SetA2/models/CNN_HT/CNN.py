from torch import nn
import torch


class Net(nn.Module):
    def __init__(self, classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)  # Convolution layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(10*6*6, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, len(classes))
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()  # Activation Function
        self.flatten = nn.Flatten()

    def set_initial_kernels(self):
        # Definito by hand i primi 5 kernels
        kernel1 = torch.tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=torch.float32)

        kernel2 = torch.tensor([[1, 0, 1],
                                [0, 1, 0],
                                [1, 0, 1]], dtype=torch.float32)

        kernel3 = torch.tensor([[1, 1, 1],
                                [0, 0, 0],
                                [1, 1, 1]], dtype=torch.float32)

        kernel4 = torch.tensor([[0, 1, 0],
                                [0, 0, 0],
                                [1, 0, 1]], dtype=torch.float32)

        kernel5 = torch.tensor([[1, 1, 0],
                                [1, 0, 0],
                                [0, 0, 0]], dtype=torch.float32)

        kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]  # list of

        for k, kernel in enumerate(kernels):
            self.conv1.weight[k, 0] = kernel

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


