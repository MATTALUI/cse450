import pandas as pd
import utils

# Load the data
bank_data = pd.read_csv("./datasets/bank.csv")
# print(bank_data.head())

# Remap some of the names to make data easier to work with
class Cols:
    age = "age"
    marital = "marital status"
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
print(bank_data.head())

# Base Report for getting to know the data
base_report = utils.BaseReport(bank_data)
# base_report.display()
