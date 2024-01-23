import pandas as pd
import utils

spam = pd.read_csv("./datasets/spam.csv")

words = "Suspicious Words"
sender = "Unknown Sender"
images = "Contains Images"
cls = "Class"

print(spam.head())

assert utils.calculate_entropy(spam, cls) == 1.0, f"{utils.calculate_entropy(spam, cls)} != 1.0"

assert utils.calculate_information_remaining(spam, words, cls) == 0.0, f"{utils.calculate_information_remaining(spam, words, cls)} != 0.0"
assert utils.calculate_information_remaining(spam, sender, cls) == 0.9182958340544896, f"{utils.calculate_information_remaining(spam, sender, cls)} != 0.9182958340544896"
assert utils.calculate_information_remaining(spam, images, cls) == 1.0, f"{utils.calculate_information_remaining(spam, images, cls)} != 1.0"

assert utils.calculate_information_gain(spam, words, cls) == 1.0, f"{utils.calculate_information_gain(spam, words, cls)} != 1.0"
assert utils.calculate_information_gain(spam, sender, cls) == 0.08170416594551044, f"{utils.calculate_information_gain(spam, sender, cls)} != 0.08170416594551044"
assert utils.calculate_information_gain(spam, images, cls) == 0.0, f"{utils.calculate_information_gain(spam, images, cls)} != 0.0"