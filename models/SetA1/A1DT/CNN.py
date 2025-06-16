from models.SetA1.SetA1 import SetA1


class A1DT(SetA1):
    """
    Class A1DT: model of the SetA1 Architecture.

    Both Convolutional and Fully Connected Layers are initialized with the default initialization. In particular,
    They're initialized with the Kaiming initialization, i.e the HE initialization.
    """
    def __init__(self):
        super().__init__()
        self.name = "DT"
        self.set = "A1"

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
