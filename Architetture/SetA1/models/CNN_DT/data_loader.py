from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def handle_dataset():
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor(), )
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor(), )

    labels_map = {
        0: 'T-shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot',
    }
    batch_size = 128  # For processing simultaneously 128 images at every weigth update

    train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True)  # For every iteration, dataset is divided into gropus of 128 samples. Shuffle helps generalizing the model

    test_dataloader = DataLoader(test_data, batch_size=batch_size)  # Same as train_dataloader but for the test

    return train_dataloader, test_dataloader, labels_map
