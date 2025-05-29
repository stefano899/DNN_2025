import os

import torch
from matplotlib import pyplot as plt
from torch import nn

from Architetture.models.SetA1.A1DT.CNN import A1DT
from Architetture.models.SetA1.A1HF.CNN import A1HF
from Architetture.models.SetA1.A1HT.CNN import A1HT
from Architetture.models.SetA2.A2DT.CNN import A2DT
from Architetture.models.SetA2.A2HF.CNN import A2HF
from Architetture.models.SetA2.A2HT.CNN import A2HT
from Architetture.test import test_loop
from data_loader import handle_dataset
from train import train_loop


def start(device):
    scelta = int(input(
        "Scegli la modalità di addestramento: \n 1- Singola: Addestra un modello alla volta \n 0- In Sequenza: "
        "Addestra tutti i modelli in sequenza \n"))

    if scelta:
        selection = int(input(
            "Scegli il modello che vuoi addestrare inserendo un numero da 1 a 6. Qui di seguito la leggenda: \n 1- "
            "A1HF, \n 2- A1DT, \n 3- A1HT,\n 4- A2HF,\n 5- A2DT, \n 6- A2HT: "))
        epochs = int(input("inserisci il numero di epoche: "))
        batch_size = int(
            input("Inserisci il batch size: "))  # For processing simultaneously 128 images at every weigth update
        begin(selection, device, epochs, batch_size)
    elif not scelta:
        print("All 6 models will be trained in sequence and all of them will have the same number of epochs: ")

        selections = [1, 2, 3, 4, 5, 6]
        epochs = int(input("inserisci il numero di epoche: "))
        batch_size = int(input("Inserisci il batch size: "))

        for selection in selections:
            begin(selection, device, epochs, batch_size)
    else:
        raise ValueError(f"Non è stata inserita un'opzione valida")

    return


def begin(selection, device, epochs, batch_size):
    train_dataloader, test_dataloader = handle_dataset(batch_size)
    model = choose_model(selection)
    # model = initialization_or_load_weights(selection)
    model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting training for {model.get_set()}{model.get_name()}")
    accuracies = []
    losses = []
    precisions = []
    f1s = []
    recalls = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch + 1, device)
        accuracy, loss, f1, precision, recall = test_loop(test_dataloader, model, loss_fn,
                                                          device, epoch)
        accuracies.append(accuracy)
        losses.append(loss)
        precisions.append(precision)
        f1s.append(f1)
        recalls.append(recall)

    plot_graphs(accuracies, losses, epochs, model, precisions, f1s, recalls)  # , precisions, f1s, recalls)
    print("-------------------------------")
    return


def hand_kernels():
    kernel1 = torch.tensor([[0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0]], dtype=torch.float32)

    kernel2 = torch.tensor([[0, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0]], dtype=torch.float32)

    kernel3 = torch.tensor([[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]], dtype=torch.float32)

    kernel4 = torch.tensor([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]], dtype=torch.float32)

    kernel5 = torch.tensor([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=torch.float32)
    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5]

    return kernels


def choose_model(name):
    """
    This function based on the input name given from the prompt will return the selected model
    """

    kernels = hand_kernels()
    if name == 1:
        model = A1HF(kernels)

    elif name == 2:
        model = A1DT()

    elif name == 3:
        model = A1HT(kernels)

    elif name == 4:
        model = A2HF(kernels)

    elif name == 5:
        model = A2DT()

    elif name == 6:
        model = A2HT(kernels)

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida: {name}")
    return model


def plot_graphs(accuracies, losses, epochs, model, precisions, f1s, recalls):  # , precisions, f1s, recalls):
    epoch_range = list(range(1, epochs + 1))

    plt.figure(figsize=(20, 12))

    # Common style settings
    plot_args = dict(marker='o', linewidth=2.5, markersize=6)

    font_title = 16
    font_label = 14
    font_tick = 12
    font_legend = 12

    # Accuracy
    plt.subplot(3, 2, 1)
    plt.plot(epoch_range, accuracies, label="Accuracy", color='blue', **plot_args)
    plt.xlabel('Epochs', fontsize=font_label)
    plt.ylabel('Accuracy', fontsize=font_label)
    plt.title('Accuracy Over Epochs', fontsize=font_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_legend)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)

    # Loss
    plt.subplot(3, 2, 2)
    plt.plot(epoch_range, losses, label="Loss", color='red', **plot_args)
    plt.xlabel('Epochs', fontsize=font_label)
    plt.ylabel('Loss', fontsize=font_label)
    plt.title('Loss Over Epochs', fontsize=font_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_legend)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)

    # Precision
    plt.subplot(3, 2, 3)
    plt.plot(epoch_range, precisions, label="Precision", color='green', **plot_args)
    plt.xlabel('Epochs', fontsize=font_label)
    plt.ylabel('Precision', fontsize=font_label)
    plt.title('Precision Over Epochs', fontsize=font_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_legend)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)

    # Recall
    plt.subplot(3, 2, 4)
    plt.plot(epoch_range, recalls, label="Recall", color='orange', **plot_args)
    plt.xlabel('Epochs', fontsize=font_label)
    plt.ylabel('Recall', fontsize=font_label)
    plt.title('Recall Over Epochs', fontsize=font_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_legend)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)

    # F1-score
    plt.subplot(3, 2, 5)
    plt.plot(epoch_range, f1s, label="F1-score", color='purple', **plot_args)
    plt.xlabel('Epochs', fontsize=font_label)
    plt.ylabel('F1-score', fontsize=font_label)
    plt.title('F1-score Over Epochs', fontsize=font_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=font_legend)
    plt.xticks(fontsize=font_tick)
    plt.yticks(fontsize=font_tick)

    # Saving plots
    plt.tight_layout()
    output_dir = f"Plots\\Set_{model.get_set()}\\{model.get_set()}{model.get_name()}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"accuracy_loss_plot_of_{model.get_set()}{model.get_name()}.png"))
    print(f"Saved plot to {output_dir}")
