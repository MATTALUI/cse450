import pandas as pd
import matplotlib.pyplot as plt
import utils

DISPLAY = False
athletes = pd.read_csv("./datasets/simpleathletes.csv")
athletes["draft"] = athletes["draft"].astype(bool)
# print(athletes.describe())

if DISPLAY:
    _fig, ax = plt.subplots()

    ax.scatter(x=athletes["speed"], y=athletes["agility"])
    ax.set_ylabel("agility")
    ax.set_xlabel("speed")
    plt.show()

print("ATHLETES TREE: [speed, agility]")
tree = utils.KDTree(athletes, ["speed", "agility"])
tree.display_text()