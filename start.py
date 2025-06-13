import os
import shutil

import torch
from matplotlib import pyplot as plt
from torch import nn

from models.SetA1.A1DT.CNN import A1DT
from models.SetA1.A1HF.CNN import A1HF
from models.SetA1.A1HT.CNN import A1HT
from models.SetA2.A2DT.CNN import A2DT
from models.SetA2.A2HF.CNN import A2HF
from models.SetA2.A2HT.CNN import A2HT
from test import test_loop

from data_loader import data_loader
from train import train_loop


def start():
    """
    Entry point for configuring and starting the training process.

    This function prompts the user to select a training mode:
        - Single mode: Train a single model;
        - Sequence mode: Train all model sequentially using the same number of epochs
        - Resume mode: Load a checkpoint and resume training from there.

    Returns:
        None
    """
    choice = int(input(
        "Choose the train mode: \n 0- Single Mode: Trains a single model \n 1- Sequence Mode: "
        "Trains all models in sequence \n 2- Load a checkpoint of a model and resume the training \n"))

    if choice == 0:

        print("Checkpoint Folder will be deleted")
        model_name = int(input(
            "Choose the name of the model inserting a number that goes from 1 to 6. Here's the legend: \n 1- "
            "A1HF, \n 2- A1DT, \n 3- A1HT,\n 4- A2HF,\n 5- A2DT, \n 6- A2HT: "))
        epochs = int(input("Insert the number of epochs: "))
        model = choose_model(model_name)
        training_testing(model, epochs)

    elif choice == 1:
        print("All 6 models will be trained in sequence and all of them will have the same number of epochs: ")

        selections = [1, 2, 3, 4, 5, 6]
        epochs = int(input("Insert the Number of Epochs: "))

        for model_name in selections:
            model = choose_model(model_name)
            training_testing(model, epochs)

    elif choice == 2:
        checkpoint_path = input("Insert the path of the checkpoint of the model that you want to resume: ")
        load_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"No valid option has been chose")

    return


def load_checkpoint(checkpoint_path):
    """
    load a checkpoint and starts the training.

    This function load a checkpoint given by the user and after instantiating the model it loads the state_dict of the
    checkpoint to continue training from where it has stopped.

    Parameters:
       checkpoint_path (string): the path of the checkpoint

    Returns:
        None
    """
    checkpoint_folder = os.path.split(checkpoint_path)[0]

    # Get the set name from the path
    set_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))

    name = os.path.basename(os.path.dirname(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)

    if set_name == 'SetA1':
        model_name = 'A1' + name
    else:
        model_name = 'A2' + name

    model_class = globals()[model_name]
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])

    shutil.rmtree(checkpoint_folder)
    epochs = checkpoint['epochs']
    epoch = checkpoint['epoch']
    remaining_epochs = epochs - epoch
    training_testing(model, remaining_epochs)
    return


def training_testing(model, epochs):
    """
    Train and evaluate a model over a specified number of epochs.

    This function sets up the training environment, including data loading, optimizer,
    and loss function. It runs the training loop, evaluates the model after each epoch,
    collects performance metrics, and at the end of the training it plots the results.

    Parameters:
        model (torch.nn.Module): The neural network model to train and evaluate.
        epochs (int): Number of training epochs.
    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 128
    train_dataloader, test_dataloader = data_loader(batch_size)
    # model = initialization_or_load_weights(selection)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting training for {model.get_set()}{model.get_name()}")
    accuracies = []
    losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        train_loop(train_dataloader, model, loss_fn, optimizer, epoch + 1, device, epochs)
        accuracy, loss = test_loop(test_dataloader, model, loss_fn,
                                   device, epoch)
        accuracies.append(accuracy)
        losses.append(loss)

    plot_graphs(accuracies, losses, epochs, model)  # , precisions, f1s, recalls)
    print("-------------------------------")
    return


def choose_model(name):
    """
    Initialization of one of the six models.

    This function, based on the input name given from the prompt, will
    return the selected model

    Returns:
        model(torch.nn.Module): the selected model.
    """

    if name == 1:
        model = A1HF()

    elif name == 2:
        model = A1DT()

    elif name == 3:
        model = A1HT()

    elif name == 4:
        model = A2HF()

    elif name == 5:
        model = A2DT()

    elif name == 6:
        model = A2HT()

    else:
        raise ValueError(f"No valid option has been chose: {name}")
    return model


def plot_graphs(accuracies, losses, epochs, model):  # , precisions, f1s, recalls):
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

    # Saving plots
    plt.tight_layout()
    output_dir = f"Plots\\Set_{model.get_set()}\\{model.get_set()}{model.get_name()}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"accuracy_loss_plot_of_{model.get_set()}{model.get_name()}.png"))
    print(f"Saved plot to {output_dir}")
