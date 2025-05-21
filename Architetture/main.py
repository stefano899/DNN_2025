
import torch
from start import start_single_mode, start_sequence_mode

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scelta = int(input("Scegli la modalità di addestramento: \n 1- Singola: Addestra un modello alla volta \n 0- In Sequenza: Addestra tutti i modelli in sequenza \n"))

    if scelta:
        start_single_mode(device)
    else:
        start_sequence_mode(device)


