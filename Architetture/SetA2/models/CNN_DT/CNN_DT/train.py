import torch
import os

def train_loop(dataloader, model, loss_fn, optimizer, epoch, device):
    size = len(dataloader.dataset)
    print(f"Training set of size: {size}")

    for batch, (X, y) in enumerate(dataloader):  # (X = input, y = target)
        X, y = X.to(device), y.to(device)  # Setting of 2 architectures

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # Loss function calculating the zero-gradient descent
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:  # every 1000 batch it prints the loss
            loss, current = loss.item(), (batch + 1) * len(X)
            current_loss = current / size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # torch save model with torch.save()
    checkpoint = {
        'epoch': epoch,  # l'epoca corrente
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Definisci il percorso della cartella dei checkpoint
    checkpoint_dir = r'C:\Users\stefa\Desktop\DNN2025\DNNStefano\DNN\SetA1\models\CNN_HF\Checkpoints'

    # Se la cartella non esiste, la creiamo
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Costruiamo il percorso completo del file checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_Model_CNN_A1_HF.pt')

    torch.save(checkpoint, checkpoint_path)


