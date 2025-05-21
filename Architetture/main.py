from Architetture.start import start
import torch
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies, losses = start(device, epochs)

    # comment if it creates problems
    matplotlib.use('TkAgg')

    epoch_range = list(range(1, epochs + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)  # Primo subplot
    plt.plot(epoch_range, accuracies, marker='o', label="Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)  # Secondo subplot
    plt.plot(epoch_range, losses, marker='o', label="Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
