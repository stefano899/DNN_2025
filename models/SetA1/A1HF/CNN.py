
from torch import nn
import torch
from custom_kernels import kernels

from models.SetA1.SetA1 import SetA1


class A1HF(SetA1):
    """
    Class A1HF: model of the SetA1 Architecture.

    First Convolutional layer is initialized with manual weights and they're not trained during the training process;
    Fully Connected Layer are initialized with the default initialization. In particular it is
    initialized with the Kaiming initialization, i.e the HE initialization.

    """
    def __init__(self):
        super().__init__()
        self.name = "HF"
        self.set = "A1"

        with torch.no_grad():
            for k, kernel in enumerate(kernels):
                self.conv1.weight[k, 0] = kernel

        for param in self.conv1.parameters():
            param.requires_grad = False

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
