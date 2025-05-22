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
    train_dataloader, test_dataloader, labels_map = handle_dataset()
    selection = int(input(
        "Scegli il modello che vuoi addestrare inserendo un numero da 1 a 6. Qui di seguito la leggenda: \n 1- A1HF, \n 2- A1DT, \n 3- A1HT,\n 4- A2HF,\n 5- A2DT, \n 6- A2HT: "))

    epochs = int(input("inserisci il numero di epoche: "))

    model = initialization_or_load_weights(selection, labels_map)
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
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch + 1, device)
        accuracy, loss, f1, precision, recall = test_loop(test_dataloader, model, loss_fn,
                                                          device)  # add them if you want more results , f1, precision, recall
        accuracies.append(accuracy)
        losses.append(loss)
        precisions.append(precision)
        f1s.append(f1)
        recalls.append(recall)

    plot_graphs(accuracies, losses, epochs, model, precisions, f1s, recalls)  # , precisions, f1s, recalls)
    print("Done!")
    return

def begin()
def start_sequence_mode(device):
    model = any

    print("All 6 models will be trained in sequence and all of them will have the same number of epochs: ")

    selections = [1, 2, 3, 4, 5, 6]

    epochs = int(input("inserisci il numero di epoche: "))
    train_dataloader, test_dataloader, labels_map = handle_dataset()

    for selection in selections:

        model = initialization_or_load_weights(selection, labels_map)
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

            accuracy, loss, f1, precision, recall = test_loop(test_dataloader, model, loss_fn,
                                                              device)  # add them if you want more results: , f1,
            # precision, recall
            accuracies.append(accuracy)
            losses.append(loss)
            precisions.append(precision)
            f1s.append(f1)
            recalls.append(recall)

        plot_graphs(accuracies, losses, epochs, model, precisions, f1s, recalls)  # , precisions, f1s, recalls)
    print("Done!")
    return


def initialization_or_load_weights(name, labels_map):
    """

    :param name: the name of the model (type = int)
    :param labels_map: the dictionary of the classes that we want to predict
    :return: returns the model

    This function is used to initialize the model and to save the initialization weights of the first convolution layer.
    In case there exists an initialization of the weights of the first convolution layer for the specified architecture,
    it applies it. Otherwise, it will initialize the model and save the weights of the first convolution layer, creating
    the initialization file of the architecture.

    Example: Suppose that you have the Architecture HT of both Sets A1 and A2. In this case, if there exists a
     weight initialization for the first conv layer of this architecture, both architectures of Sets A1 and A2 will
     have the same weights in the first convolution layer.

    """
    model = choose_model(name, labels_map)
    init_dir = f"Initializations\\{model.get_name()}"
    weights_path = os.path.join(init_dir, f"{model.get_name()}.initialization.pth")

    if not os.path.exists(init_dir):
        os.makedirs(init_dir, exist_ok=True)
        print("Initialization path doesn't exist, i'm gonna save its first weights initialization.")
        conv1_state_dict = {"conv1.weight": model.conv1.weight.data.clone(),
                            "conv1.bias": model.conv1.bias.data.clone()}
        torch.save(conv1_state_dict, weights_path)
        print(f"Initialization saved in {weights_path}. From now on, every new training process involving {model.get_name()} "
              f"(independent from the set) will be instantiated with these weights.")
    else:
        print(f"There exists a weight initialization file for {model.get_name()}. I'm going to apply it to the model")
        conv1_weights = torch.load(weights_path)
        with torch.no_grad():
            model.conv1.weight.copy_(conv1_weights["conv1.weight"])
            model.conv1.bias.copy_(conv1_weights["conv1.bias"])

        # DEBUG For checking if it applies the weights in a correct way.
        # Confronta i pesi del primo layer
        #init = torch.load(weights_path)
        #print(init.keys())
        #layer_name = "conv1.weight"  # Modifica con il nome corretto del primo layer nel tuo modello
        #weights_folder = init[layer_name]
        #weights_model = model.conv1.weight.data
        #print(f"weights_folder:")
        #print(weights_model)
        #print(f"weights_folder:")
        #print(weights_folder)
        #if torch.allclose(weights_model, weights_folder, atol=1e-6):
        #    print("I pesi iniziali del primo layer sono coerenti tra le reti!")
        #else:
        #    print("Attenzione! I pesi iniziali sono diversi.")

    return model


def choose_model(name, labels_map):
    """
    This function based on the input name given from the prompt will return the selected model
    :param name: input selected from prompt
    :param labels_map: dictionary of the classes that we want to predict
    :return: the model
    """
    if name == 1:
        model = A1DT(labels_map)

    elif name == 2:
        model = A1HF(labels_map)

    elif name == 3:
        model = A1HT(labels_map)

    elif name == 4:
        model = A2DT(labels_map)

    elif name == 5:
        model = A2HF(labels_map)

    elif name == 6:
        model = A2HT(labels_map)

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida: {name}")
    return model


def plot_graphs(accuracies, losses, epochs, model, precisions, f1s, recalls):  # , precisions, f1s, recalls):
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
