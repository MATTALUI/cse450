import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


housing = pd.read_csv("./datasets/housing.csv")
print(housing.head())
ignored_features = ["id", "date", "price"]

target="price"

x = housing.drop(ignored_features, axis=1)
y = housing[target]

################################################################################
# BASE CASE                                                                    #
################################################################################
run_count = 5
scores = []
for i in range(run_count):
    seed= 69 * (i + 1)
    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=seed
    )
    # Build classifier
    gbr = GradientBoostingRegressor(max_depth=5, random_state=seed)
    # Train
    gbr.fit(x_train, y_train)
    # Make Predictions
    predictions = gbr.predict(x_test)
    # Evaluate the model
    score = r2_score(y_test, predictions)
    scores.append(score)
print("Base Case")
print("score: ", scores)
print("average: ", np.average(scores))