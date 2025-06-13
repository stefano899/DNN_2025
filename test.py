import os

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def test_loop(dataloader, model, loss_fn, device, epoch):
    """
      Evaluates the model on the test dataset and logs performance metrics.

      This function runs tests on the test dataset and logs performance metrics,

      Parameters:
          :dataloader: The DataLoader providing the test data.
          :model: The model to evaluate. Must implement `get_name()` and `get_set()`.
          :loss_fn: The loss function used for evaluation.
          :device: The device (CPU or GPU) to run inference on.
          :epoch: The current epoch number (used for logging).

      Returns:
          tuple:
              - accuracy: Overall accuracy of the model on the test set.
              - test_loss: Average loss over the test dataset.
              - f1: F1 score.
              - precision: precision score.
              - recall: recall score.
      """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted = pred.argmax(1)  # assumes output is logits or probabilities over classes
            correct += (predicted == y).sum().item()
            total += y.size(0)

    test_loss /= num_batches
    accuracy = 100 * correct / total  # convert to percentage
    with open(f"logs\\{model.get_set()}{model.get_name()}_test_logs.txt", "a") as f:
        f.write(f"EPOCH: {epoch}. \n \n Accuracy: {accuracy:>8f}, Avg loss: {test_loss:>8f} \n")
    print(
        f"Test Error: \n Accuracy: {accuracy:>8f}%, Avg loss: {test_loss:>8f}%")
    return accuracy, test_loss  # , f1, precision, recall
