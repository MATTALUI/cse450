import pandas as pd
import utils

titanic = pd.read_csv("./datasets/titanicreading4.csv")
titanic["survived"] = titanic["survived"].astype(bool)

report = utils.BaseReport(titanic)
report.display()