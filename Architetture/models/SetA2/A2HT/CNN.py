from torch import nn
import torch
from torch.nn import init

from Architetture.models.SetA2.SetA2 import SetA2


class A2HT(SetA2):
    def __init__(self, kernels):
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
