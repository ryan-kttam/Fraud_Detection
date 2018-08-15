# Credit Card Fraud Detection

The goal of this project is to develop a supervised machine learning model to identify frauds. This project will also discuss about the advantages and disadvantages regarding accuracy, recall, and precision. 

## Getting Started 
This project requires Python 2 and the following packages:
```
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score
from sklearn.svm import SVC
```
## Data Exploration
Before we develop any machine learning model, we have to get ourselves to be familiar with the data. Due to privacy issues, the data provider is not able to provide actual features. Alternatively, the data provider performed PCA on most of the features (excluding 'Time', 'Class', and 'Amount'.
 - Time is the number of seconds elapsed between this transaction and the first transac tion in the dataset. 
 - Class is the classification: 1 if this transaction is a fraud, 0 otherwise. 
 - Amount is the number of dollar spent in a transaction.
```
# 284807 rows * 31 columns
dataset.shape
# print out the first 5 rows of the dataset
dataset.head()
#print out the summary for each column (mean, std, median, etc)
dataset.describe()
```
```
# Count how many cases that are identified as fraud
from collections import Counter
fraud_distribution = Counter(dataset['Class'])
# there are less than 0.2% cases that are fraud
len(dataset[dataset['Class'] == 1])* 100.0 / len(dataset)
```
This dataset is very imbalance; Out of 284,807 records, there are only 0.17% fraud cases. This is optimal for the company, but it might not necessary good for training a machine learning model. 
