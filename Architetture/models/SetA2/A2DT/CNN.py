from torch import nn
import torch
from torch.nn import init

from Architetture.models.SetA2.SetA2 import SetA2


class A2DT(SetA2):
    def __init__(self):
        super().__init__()
        self.name = "DT"
        self.set = "A2"

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
