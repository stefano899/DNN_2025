import torch
import os


def train_loop(dataloader, model, loss_fn, optimizer, epoch, device, epochs):
    """
    Perform the training epoch on the given model using the provided dataloader

    This function iterates over the training dataset, computes the loss, performs
    backpropagation, updates model weights, logs loss every 1000 batches, and saves
    a model checkpoint at the end of the epoch.


     Parameters:
        :param dataloader: The DataLoader providing training batches.
        :param model: The model to train. Must implement `get_name()` and `get_set()`.
        :param loss_fn: The loss function used to compute the training loss.
        :param optimizer: The optimizer for updating model parameters.
        :param epoch: The current epoch number (0-indexed).
        :param device: The device (CPU or GPU) on which training is performed.
        :param epochs: Total number of epochs (used for checkpoint metadata).

    Returns:
        None

    """
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
            if not os.path.exists('./logs'):
                os.makedirs('./logs')
            with open(f"logs\\{model.get_set()}{model.get_name()}_train_logs.txt", "a") as f:
                f.write(
                    f" EPOCH: {epoch} \n loss: {loss:>7f} "
                    f" [{current:>5d}/{size:>5d}] \n \n")

    # torch save model with torch.save()
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'epoch': epoch
    }

    # Impose the directory of where you want to save checkpoints
    checkpoint_dir = f'Checkpoints\\Set{model.get_set()}\\{model.get_name()}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir,
                                   f'epoch_{epoch + 1}_Model_CNN_{model.get_name()}.pth')
    print(f"Checkpoint saved to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)

    return
