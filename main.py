import os
import shutil
from start import start

if __name__ == "__main__":

    folders = ["Plots", "logs"]
    decision = input(
        "Warning: once main is executed, Plots and Logs folders will be eliminated and then recreated. \n If you want "
        "to retrain"
        " a model, please be sure that you have deleted the corresponding checkpoint folder."
        " \n PROCEED? Y/n ")

    if decision == "Y" or decision == "y":
        for folder in folders:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"{folder} was deleted.")
            else:
                print(f"{folder} doesn't exists.")
        start()

    elif decision == "n" or decision == "N":
        print("Goodbye")

    else:
        raise ValueError(f"Non è stata inserita un'opzione valida")


