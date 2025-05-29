from torch import nn
import torch
from torch.nn import init

from Architetture.models.SetA1.SetA1 import SetA1


# 5*14*14 * 10 + 3*3*5 = 9845

class A1DT(SetA1):

    def __init__(self):
        super().__init__()
        self.name = "DT"
        self.set = "A1"

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
