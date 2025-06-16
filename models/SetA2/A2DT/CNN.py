from models.SetA2.SetA2 import SetA2


class A2DT(SetA2):
    """
    Class A2DT: model of the SetA2 Architecture.

    Both Convolutional and Fully Connected Layers are initialized with the default initialization. In particular,
    they're initialized with the Kaiming initialization, i.e the HE initialization.
    """
    def __init__(self):
        super().__init__()
        self.name = "DT"
        self.set = "A2"

    def get_name(self):
        return self.name

    def get_set(self):
        return self.set
