from torch import nn
import torch
from torch.nn import init

from Architetture.models.SetA1.SetA1 import SetA1


class A1HF(SetA1):
    def __init__(self, kernels):
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
