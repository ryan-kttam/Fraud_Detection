import pandas as pd
import numpy as np

dataset = pd.read_csv('creditcard.csv')

# print our the dimension of the dataset
# 284807 rows * 31 columns
dataset.shape

# print out the first 5 rows of the dataset
dataset.head()

#print out the summary (mean, std, median, etc)
dataset.describe()

# Count how many cases that are identified as fraud
from collections import Counter
Counter(dataset['Class'])
# or sum(dataset['Class'])

# there are less than 0.2% cases that are fraud
len(dataset[dataset['Class'] == 1])* 100.0 / len(dataset)
