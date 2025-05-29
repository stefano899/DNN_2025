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
    train_dataloader, test_dataloader, labels_map = handle_dataset(batch_size)
    model = initialization_or_load_weights(selection, labels_map)
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
                                                          device)  # add them if you want more results , f1, precision, recall
        accuracies.append(accuracy)
        losses.append(loss)
        precisions.append(precision)
        f1s.append(f1)
        recalls.append(recall)

    plot_graphs(accuracies, losses, epochs, model, precisions, f1s, recalls)  # , precisions, f1s, recalls)
    print("-------------------------------")
    return



def initialization_or_load_weights(name, labels_map):
    model = choose_model(name, labels_map)
    init_dir = "Initializations"

    # Create subfolder for model name
    weights_conv1_dir = os.path.join(init_dir, model.get_name())
    weights_conv1_path = os.path.join(weights_conv1_dir, f"{model.get_name()}_conv1_weights_initialization.pth")

    # Create subfolder for model set
    other_weights_dir = os.path.join(init_dir, model.get_set())
    other_weights_path = os.path.join(other_weights_dir, "fc1_and_or_conv2_weights_initialization.pth")

    if not os.path.exists(weights_conv1_path):
        os.makedirs(weights_conv1_dir, exist_ok=True)
        print("Initialization conv1 path doesn't exist, I'm going to save its first weights initialization.")
        conv1_state_dict = {
            "conv1.weight": model.conv1.weight.data.clone(),
            "conv1.bias": model.conv1.bias.data.clone()
        }
        torch.save(conv1_state_dict, weights_conv1_path)
    else:
        print(f"There exists a conv1 weight initialization file for {model.get_set()}{model.get_name()}. I'm going to apply it to the model.")
        conv1_weights = torch.load(weights_conv1_path)
        with torch.no_grad():
            model.conv1.weight.copy_(conv1_weights["conv1.weight"])
            model.conv1.bias.copy_(conv1_weights["conv1.bias"])

    if not os.path.exists(other_weights_path):
        print("Initialization other layers path doesn't exist, I'm going to save its first weights initialization.")
        os.makedirs(other_weights_dir, exist_ok=True)
        if model.get_set() == "A1":
            fc1_state_dict = {
                "fc1.weight": model.fc1.weight.data.clone(),
                "fc1.bias": model.fc1.bias.data.clone()
            }
            torch.save(fc1_state_dict, other_weights_path)
        else:
            fc2_conv2_state_dict = {
                "conv2.weight": model.conv2.weight.data.clone(),
                "conv2.bias": model.conv2.bias.data.clone(),
                "fc1.weight": model.fc1.weight.data.clone(),
                "fc1.bias": model.fc1.bias.data.clone()
            }
            torch.save(fc2_conv2_state_dict, other_weights_path)

        print(
            f"Initialization saved in {other_weights_path}.")

    else:
        print(f"There exists other layers initialization file for {model.get_set()}{model.get_name()}."
              f"\n I'm going to apply it to the model.")

        if model.get_set() == "A1":
            fc1_weights = torch.load(other_weights_path)
            with torch.no_grad():
                model.fc1.weight.copy_(fc1_weights["fc1.weight"])
                model.fc1.bias.copy_(fc1_weights["fc1.bias"])
        else:
            conv2_fc1_weights = torch.load(other_weights_path)
            with torch.no_grad():
                model.conv2.weight.copy_(conv2_fc1_weights["conv2.weight"])
                model.conv2.bias.copy_(conv2_fc1_weights["conv2.bias"])
                model.fc1.weight.copy_(conv2_fc1_weights["fc1.weight"])
                model.fc1.bias.copy_(conv2_fc1_weights["fc1.bias"])

    # DEBUG For checking if it applies the weights in a correct way.
    # Confronta i pesi del primo layer
    #init = torch.load(weights_conv1_path)
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
        model = A1HF(labels_map)

    elif name == 2:
        model = A1DT(labels_map)

    elif name == 3:
        model = A1HT(labels_map)

    elif name == 4:
        model = A2HF(labels_map)

    elif name == 5:
        model = A2DT(labels_map)

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
