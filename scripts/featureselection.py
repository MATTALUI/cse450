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

scaler = MinMaxScaler()

housing["has_been_renovated"] = housing["yr_renovated"].map(lambda v: 1 if v > 0 else v)
x = housing.drop(ignored_features, axis=1)
# x[normalized_columns] = scaler.fit_transform(x[normalized_columns])
# x["date"] = 
# .strptime('20150423T000000','%Y%m%dT%H%M%S%f').timestamp()
y = housing[target]
# print(x.head(25))

################################################################################
# BASE CASE                                                                    #
################################################################################
run_count = 10
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
print("average: ", np.average(scores)) # 0.8793779043184102
print("Improved: ", np.average(scores) > 0.8793779043184102) #R2 value to beat. You've done better when this is true

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
