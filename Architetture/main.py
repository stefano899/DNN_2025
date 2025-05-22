import os
import shutil

import torch
from start import start

if __name__ == "__main__":

    # Imposta il seme globalmente PRIMA di qualsiasi operazione casuale
    seed = 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    folders = ["Checkpoints", "Plots"]
    decisione = input(
        "Warning: una volta fatto partire il main verranno cancellate le cartelle Checkpoints e Plots (per poi essere ricreate) PROCEDERE? S/n ")

    if decisione == "S":
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"{folder} was deleted.")
            else:
                print(f"{folder} doesn't exists.")

    elif decisione == "n":
        exit()

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start(device)