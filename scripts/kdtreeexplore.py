import pandas as pd
import matplotlib.pyplot as plt
import utils

DISPLAY = False
athletes = pd.read_csv("./datasets/simpleathletes.csv")
athletes["draft"] = athletes["draft"].astype(bool)

if DISPLAY:
    _fig, ax = plt.subplots()

    ax.scatter(x=athletes["speed"], y=athletes["agility"])
    ax.set_ylabel("agility")
    ax.set_xlabel("speed")
    plt.show()

print("ATHLETES TREE: [speed, agility]")
tree = utils.KDTree(athletes, ["speed", "agility"])
tree.display_text()
assert tree.to_depth_first_array() == [6, 3, 7, 8, 11, 9, 10, 4, 5, 1, 2, 16, 21, 15, 12, 20, 18, 19, 17, 14, 13]

pokemon = pd.read_csv("./datasets/pokemon.csv")
pokemon = pokemon[["number", "height", "weight"]][:6]
print("POKEMON TREE: [height, weight]")
poke_tree = utils.KDTree(pokemon, ["height", "weight"], identifying_feature="number")
poke_tree.display_text()
assert poke_tree.to_depth_first_array() == [5.0, 4.0, 1.0, 2.0, 3.0, 6.0]