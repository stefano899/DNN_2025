import torch
from torch import nn

from Architetture.SetA1.models.A1DT.CNN import A1DT
from Architetture.SetA1.models.A1HF.CNN import A1HF
from Architetture.SetA1.models.A1HT.CNN import A1HT
from Architetture.SetA2.models.A2DT.CNN import A2DT
from Architetture.SetA2.models.A2HF.CNN import A2HF
from Architetture.SetA2.models.A2HT.CNN import A2HT
from Architetture.data_loader import handle_dataset
from Architetture.test import test_loop
from Architetture.train import train_loop


def start(device, epochs):
    name = input("Digita il nome della rete che vuoi addestrare: A1HF, A1DT, A1HT, A2HF, A2DT, A2HT (copia uno dei seguenti nomi ed incollalo di fianco): ").strip()
    model = any
    train_dataloader, test_dataloader, labels_map = handle_dataset()

    if name == "A1DT":
        model = A1DT(labels_map)

    elif name == "A1HF":
        model = A1HF(labels_map)

    elif name == "A1HT":
        model = A1HT(labels_map)

    elif name == "A2DT":
        model = A2DT(labels_map)

    elif name == "A2HF":
        model = A2HF(labels_map)

    elif name == "A2HT":
        model = A2HT(labels_map)

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida: {name}")

    model.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting training for {model.name}")
    accuracies = []
    losses = []

    for iterator in range(epochs):
        print(f"Epoch {iterator + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, iterator + 1, device)
        accuracy, loss = test_loop(test_dataloader, model, loss_fn, device)
        accuracies.append(accuracy)
        losses.append(loss)
    print("Done!")

    return accuracies, losses
