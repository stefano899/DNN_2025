from DNNStefano.DNN.SetA1.models.CNN_HF.data_loader import handle_dataset
from DNNStefano.DNN.SetA1.models.CNN_HF.test import test_loop
from DNNStefano.DNN.SetA1.models.CNN_HF.train import train_loop
from DNNStefano.DNN.SetA1.models.CNN_HF.CNN import Net
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt


def start(epochs, iteratore, device, train_loader, test_loader):
    for iterator in range(iteratore, epochs):
        print(f"Epoch {iterator + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, iterator + 1, device)
        accuracy, loss = test_loop(test_dataloader, model, loss_fn, device)
        accuracies.append(accuracy)
        losses.append(loss)
    print("Done!")


if __name__ == "__main__":
    train_dataloader, test_dataloader, labels_map = handle_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(labels_map)  # Dichiarazione di un oggetto di tipo Net
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    model.set_initial_kernels()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 20
    accuracies = []
    losses = []
    iterator = 0
    start(epochs, iterator, device, train_dataloader, test_dataloader)

    epochs = range(1, len(accuracies) + 1)

    # comment if it creates problems
    matplotlib.use('TkAgg')  # Oppure 'Qt5Agg' se hai PyQt5 installato

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # Primo subplot
    plt.plot(epochs, accuracies, marker='o', label="Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)  # Secondo subplot
    plt.plot(epochs, losses, marker='o', label="Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
