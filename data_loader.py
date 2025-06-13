from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def data_loader(batch_size):
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor(), )
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor(), )
    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=False)  # For every iteration, dataset is divided into gropus of 128 samples.
    test_dataloader = DataLoader(test_data, batch_size=batch_size)  # Same as train_dataloader but for the test
    return train_dataloader, test_dataloader
