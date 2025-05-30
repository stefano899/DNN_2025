from torch import nn
import torch
from torch.nn import init

from models.SetA2.SetA2 import SetA2


class A2HF(SetA2):
    def __init__(self, kernels):
        super(A2HF, self).__init__()
        self.name = "HF"
        self.set = "A2"

        with torch.no_grad():
            for k, kernel in enumerate(kernels):
                self.conv1.weight[k, 0] = kernel

        # freezing first layer
        for param in self.conv1.parameters():
            param.requires_grad = False

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
