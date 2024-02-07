import pandas as pd
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load the data
bank_data = pd.read_csv("./datasets/bank.csv")
# print(bank_data.head())

# Remap some of the names to make data easier to work with
class Cols:
    age = "age"
    marital = "marital_status"
    education = "education"
    default = "has_credit_in_default"
    housing = "has_housing_loan"
    loan = "has_personal_loan"
    # last contact of current campaign
    contact = "last_contact_method"
    month = "last_contact_month"
    dayofweek = "last_contact_weekday"
    # Info for previous campaigns
    campaign = "total_campaign_contacts"
    pdays = "days_since_last_contact"
    previous = "total_contacts"
    poutcome = "previous_outcome"
    evr = "employment_variation_rate"
    cpi = "consumer_price_index"
    cci = "consumer_confidence_index"
    euribor = "euribor_rate"
    employees = "number_of_employees"
    subscribed = "has_subscribed"
name_map = {
    "marital": Cols.marital,
    "default": Cols.default,
    "housing": Cols.housing,
    "loan": Cols.loan,
    "contact": Cols.contact,
    "month": Cols.month,
    "dayofweek": Cols.dayofweek,
    "campaign": Cols.campaign,
    "pdays": Cols.pdays,
    "previous": Cols.previous,
    "poutcome": Cols.poutcome,
    "emp.var.rate": Cols.evr,
    "cons.price.idx": Cols.cpi,
    "cons.conf.idx": Cols.cci,
    "euribor3m": Cols.euribor,
    "nr.employed": Cols.employees,
    "y": Cols.subscribed
}
bank_data = bank_data.rename(columns=name_map)
# print(bank_data.head())

# Base Report for getting to know the data
base_report = utils.BaseReport(bank_data)
# print(base_report.continuous_features)
# for col in base_report.categorical_features:
#     if "unknown" in base_report.dataframe[col].unique():
#         print(
#             col,
#             len(base_report.dataframe[base_report.dataframe[col] == "unknown"]),
#             len(base_report.dataframe),
#             len(base_report.dataframe[base_report.dataframe[col] == "unknown"]) / len(base_report.dataframe) * 100
#         )
# base_report.dataframe.replace(to_replace="services", value=None, inplace=True)
# base_report.display()

small_bank = bank_data.head(10)
print(small_bank.head(10))
# print(small_bank[Cols.marital].unique().tolist().index())
small_bank[Cols.marital] = small_bank[Cols.marital].map(lambda v: small_bank[Cols.marital].unique().tolist().index(v))
print(small_bank.head(10))

# features_we_care_about = [Cols.default, Cols.housing, Cols.loan]
# targets = [Cols.subscribed]

# x = pd.get_dummies(bank_data[features_we_care_about], dtype=int)
# y = bank_data[targets].map(lambda t: { "yes": 1, "no": 0}[t])
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

# classifier = tree.DecisionTreeClassifier()
# classifier = classifier.fit(x_train, y_train)

# predictions = classifier.predict(x_test)
# score = accuracy_score(y_test, predictions)
# print(predictions)
# print(score)

