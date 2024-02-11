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
# print(housing.head())
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
# plt.scatter(x=housing['long'], y=housing['lat'])
# plt.show()

scaler = MinMaxScaler()
x=housing
# housing["has_been_renovated"] = housing["yr_renovated"].map(lambda v: 1 if v > 0 else v)
x["years_since_renovation"] = datetime.today().year - x[["yr_renovated", "yr_built"]].max(axis=1)
x = x.drop(ignored_features, axis=1)
# bins = 100
# x["lat"] = pd.cut(x["lat"], bins=bins, labels=range(bins))
# x["long"] = pd.cut(x["long"], bins=bins, labels=range(bins))
# x[normalized_columns] = scaler.fit_transform(x[normalized_columns])
# x["date"] = 
# .strptime('20150423T000000','%Y%m%dT%H%M%S%f').timestamp()
y = housing[target]

################################################################################
# BASE CASE                                                                    #
################################################################################
run_count = 10
scores = []
known_best = 0.8800140509386003 #R2 value to beat.
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
