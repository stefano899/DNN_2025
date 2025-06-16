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

MODELS_DICTIONARY = {
    'A1HF': A1HF,
    'A1HT': A1HT,
    'A1DT': A1DT,
    'A2HF': A2HF,
    'A2HT': A2HT,
    'A2DT': A2DT
}


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
        model_name = input(
            "Choose the name of the model by copying one of the following model names into the prompt: \n "
            "A1HF, \n A1DT, \n A1HT,\n A2HF,\n A2DT, \n A2HT: ")
        epochs = int(input("Insert the number of epochs: "))

        if model_name in MODELS_DICTIONARY:
            model = MODELS_DICTIONARY[model_name]()
            training_testing(model, epochs)
        else:
            raise ValueError(f"{model_name} is not a valid model")

    elif choice == 1:
        print("All 6 models will be trained in sequence and all of them will have the same number of epochs: ")
        epochs = int(input("Insert the Number of Epochs: "))

        for key in MODELS_DICTIONARY:
            model = MODELS_DICTIONARY[key]()
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
    checkpoint to continue training from where it has stopped. The path of the checkpoint must be of this format:
    ../SetA1/DT/epoch_2_Model_CNN_DT.pth

    Parameters:
       checkpoint_path (string): the path of the checkpoint

    Returns:
        None
    """

    checkpoint = torch.load(checkpoint_path)

    checkpoint_folder = os.path.split(checkpoint_path)[0]

    # Get the set name from the path
    set_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
    # Get the model name from the path
    name = os.path.basename(os.path.dirname(checkpoint_path))

    if set_name == 'SetA1':
        model_name = 'A1' + name
    else:
        model_name = 'A2' + name

    if model_name in MODELS_DICTIONARY:
        model = MODELS_DICTIONARY[model_name]()
        model.load_state_dict(checkpoint['model_state_dict'])
        shutil.rmtree(checkpoint_folder)
        epochs = checkpoint['epochs']
        epoch = checkpoint['epoch']
        remaining_epochs = epochs - epoch
        training_testing(model, remaining_epochs)
    else:
        raise ValueError(f"{model_name} not found or the checkpoint doesn't exists")

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


def plot_and_save(epoch_range, y_values, y_label, title, color, filepath):
    """
    Create and Saves Plots from a model.

    This function creates a plot and saves it in the specified path.

    Parameters:
        epoch_range (list of int): List of epoch numbers (e.g., [1, 2, 3, ..., n]) representing x-axis values.
        y_values (list of float): Corresponding metric values for each epoch on the y-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        color (str): Color name for the plot line.
        filepath (str): path where to save the plot image
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, y_values, label=y_label, color=color, marker='o', linewidth=2.5, markersize=6)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {y_label.lower()} plot to {filepath}")


def plot_graphs(accuracies, losses, epochs, model):
    """
    Parameters:
        accuracies (list): Accuracy values for each epoch.
        losses (list): Loss values for each epoch.
        epochs (int): Total number of training epochs.
        model (object): model of the Architecture.
    """

    epoch_range = list(range(1, epochs + 1))
    output_dir = os.path.join("Plots", f"Set{model.get_set()}", f"{model.get_set()}{model.get_name()}")

    os.makedirs(output_dir, exist_ok=True)

    accuracy_path = os.path.join(output_dir, f"accuracy_plot_of_{model.get_set()}{model.get_name()}.png")
    loss_path = os.path.join(output_dir, f"loss_plot_of_{model.get_set()}{model.get_name()}.png")

    plot_and_save(epoch_range, accuracies, 'Accuracy', 'Accuracy Over Epochs', 'blue', accuracy_path)
    plot_and_save(epoch_range, losses, 'Loss', 'Loss Over Epochs', 'red', loss_path)
