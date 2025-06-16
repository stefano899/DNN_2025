import torch
from custom_kernels import kernels

from models.SetA2.SetA2 import SetA2


class A2HT(SetA2):
    """
    Class A2HT: model of the SetA1 Architecture.

    First Convolutional layer is initialized with manual weights, and it will be trained during the training process;
    Second Convolutional Layer and the Fully Connected Layer are initialized with the default initialization.
    In particular, they're initialized with the Kaiming initialization, i.e the HE initialization.

    """
    def __init__(self):
        super().__init__()
        self.name = "HT"
        self.set = "A2"

        with torch.no_grad():
            for k, kernel in enumerate(kernels):
                self.conv1.weight[k, 0] = kernel

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
