import torch
from custom_kernels import kernels

from models.SetA1.SetA1 import SetA1


class A1HT(SetA1):
    def __init__(self):
        super().__init__()
        self.name = "HT"
        self.set = "A1"

        # Set initial weights for conv1
        with torch.no_grad():
            for k, kernel in enumerate(kernels):
                self.conv1.weight[k, 0] = kernel

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
