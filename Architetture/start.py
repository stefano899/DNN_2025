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
from data_loader import handle_dataset
from train import train_loop
from test import test_loop


def start_single_mode(device):
    model = any

    name = input(
        "Scegli il nome del modello che vuoi addestrare. Copia uno di questi nomi e incollali di fianco: A1HF, A1DT, A1HT, A2HF, A2DT, A2HT: ")

    epochs = int(input("inserisci il numero di epoche: "))
    train_dataloader, test_dataloader, labels_map = handle_dataset()

    if name == "A1DT":
        model = A1DT(labels_map)

    elif name == "A1HF":
        model = A1HF(labels_map)
        model.set_initial_kernels()

    elif name == "A1HT":
        model = A1HT(labels_map)
        model.set_initial_kernels()
    elif name == "A2DT":
        model = A2DT(labels_map)

    elif name == "A2HF":
        model = A2HF(labels_map)
        model.set_initial_kernels()
    elif name == "A2HT":
        model = A2HT(labels_map)
        model.set_initial_kernels()

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida: {name}")

    model.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting training for {model.name}")
    accuracies = []
    losses = []
    precisions = []
    f1s = []
    recalls = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch + 1, device, epochs)
        accuracy, loss = test_loop(test_dataloader, model, loss_fn,
                                   device)  # add them if you want more results , f1, precision, recall
        accuracies.append(accuracy)
        losses.append(loss)
        # precisions.append(precision)
        # f1s.append(f1)
        # recalls.append(recall)

    plot_graphs(accuracies, losses, epochs, model)  # , precisions, f1s, recalls)
    print("Done!")
    return


def start_sequence_mode(device):
    model = any

    print("All 6 models will be trained in sequence and all of them will have the same number of epochs: ")

    names = {"A1HF",
             "A1DT",
             "A1HT",
             "A2HF",
             "A2DT",
             "A2HT"}

    epochs = int(input("inserisci il numero di epoche: "))
    train_dataloader, test_dataloader, labels_map = handle_dataset()

    for name in names:
        if name == "A1DT":
            model = A1DT(labels_map)


        elif name == "A1HF":
            model = A1HF(labels_map)
            model.set_initial_kernels()
        elif name == "A1HT":
            model = A1HT(labels_map)
            model.set_initial_kernels()
        elif name == "A2DT":
            model = A2DT(labels_map)

        elif name == "A2HF":
            model = A2HF(labels_map)
            model.set_initial_kernels()
        elif name == "A2HT":
            model = A2HT(labels_map)
            model.set_initial_kernels()
        else:
            raise ValueError(f"Non è stata inserita un'opzione valida: {name}")

        model.to(device)
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        print(f"Starting training for {model.name}")
        accuracies = []
        losses = []
        precisions = []
        f1s = []
        recalls = []

        for iterator in range(epochs):
            print(f"Epoch {iterator + 1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, iterator + 1, device)
            accuracy, loss = test_loop(test_dataloader, model, loss_fn,
                                       device)  # add them if you want more results: , f1, precision, recall
            accuracies.append(accuracy)
            losses.append(loss)
            # precisions.append(precision)
            # f1s.append(f1)
            # recalls.append(recall)

        plot_graphs(accuracies, losses, epochs, model)  # , precisions, f1s, recalls)
    print("Done!")
    return


def plot_graphs(accuracies, losses, epochs, model):  # , precisions, f1s, recalls):
    epoch_range = list(range(1, epochs + 1))

    plt.figure(figsize=(20, 12))  # Più grande per chiarezza

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

    plt.tight_layout()
    # Saving plots
    output_dir = f"Plots\\{model.get_name()}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"accuracy_loss_plot_of_{model.get_name()}.png"))
    print(f"Saved plot to {output_dir}")


"""
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
"""
