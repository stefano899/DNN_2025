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

        with torch.no_grad():
            # I pesi di conv1 hanno shape [5, 1, 3, 3] (5 kernels, 1 canale, dimensione 3x3)
            # Assegno ciascun pattern al corrispondente kernel per l'unico canale in ingresso.
            for k, kernel in enumerate(kernels):
                self.conv1.weight[k, 0] = kernel



    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))

        x = self.flatten(x)  # x becomes a 2D vector

        x = self.relu(self.fc1(x))  # actv. function applied on the first fully connected layer of output

        x = self.fc2(x)  # Prediction
        return x


