import pandas as pd
import utils

pokemon = pd.read_csv("./datasets/pokemon.csv")

# Noticed that the pokemon's weights were being recorded without the proper
# precision, so  they were recording at 10x heaver than they truly are. Simply
# dividing the column by 10 should do the trick
pokemon["weight"] = pokemon["weight"] / 10.0
# Add a calculated total base stat column
pokemon["total_base_stats"] = pokemon["hp"] + pokemon["speed"] + pokemon["attack"] + pokemon["defense"] + pokemon["special-attack"] + pokemon["special-defense"]
# type2 column has many missing values. So instead we can use a proxy with no
# misisng values to indicate similar ideas
pokemon["is_dual_type"] = ~pokemon["type2"].isnull()

report = utils.BaseReport(pokemon)
report.display()
# report.write_splot("pokemon")