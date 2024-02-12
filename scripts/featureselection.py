import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

housing = pd.read_csv("./datasets/housing.csv")
target="price"
ignored_features = [
    target, #the target
    "id", # useless
    "date", # Don't know how to normalize

    "yr_renovated",
    # "lat",
    # "long",
]
normalized_columns = [
    "bedrooms",
    "bathrooms",
    "floors",
    "sqft_living",
    "sqft_lot",
]

scaler = MinMaxScaler()
housing["date"] = pd.to_datetime(housing["date"])
housing=housing.sort_values('date')
housing=housing.drop_duplicates("id", keep="last")
housing=housing.sort_index() # Maintains ATE for our experiments
housing["years_since_renovation"] = datetime.today().year - housing[["yr_renovated", "yr_built"]].max(axis=1)
housing["time_on_market"] = housing["date"].map(lambda d: datetime.today().year - d.year)

x = housing.drop(ignored_features, axis=1)
y = housing[target]

################################################################################
# BASE CASE                                                                    #
################################################################################
run_count = 10
scores = []
known_best = 0.8780341558471283 #R2 value to beat.
for i in range(run_count):
    seed= 69 * (i + 1)
    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
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
average_score = np.average(scores)
print("Base Case")
print("Score: ", average_score)
print("High Score: ", known_best)
if average_score > known_best:
    print("YOU IMPROVED IT!")
if average_score < known_best:
    print("You made it worse")
if average_score == known_best:
    print("No change")


################################################################################
# VARIANCE THRESHOLD                                                           #
################################################################################
run_count = 5
scores = []
for i in range(run_count):
    break
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
# print("Variance Threshold")
# print("score: ", scores)
# print("average: ", np.average(scores))
