import os
import shutil
from start import start

if __name__ == "__main__":

    folders = ["Plots", "logs"]
    decision = input(
        "Warning: once main is executed, Plots and logs will be eliminated and then recreated. \n If you want to retrain"
        "a model, Please be sure that you have deleted the corresponding checkpoint folder."
        " PROCEED? S/n ")

    if decision == "S" or decision == "s":
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"{folder} was deleted.")
            else:
                print(f"{folder} doesn't exists.")

    elif decision == "n" or decision == "N":
        exit()

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida")

    start()
