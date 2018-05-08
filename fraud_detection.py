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

# select all features except the response variable
# using iloc: [row, col], here, I selected all rows, and 0:-1 columns
features = dataset.iloc[:, 0:-1]
response = dataset['Class']

# explore features' relationship of each other
# first 5 features
# V1 has a positive relationship between V2 and V3, while a negative relationship with V4
# all V1 to V4 have a very sharp density at 0
pd.scatter_matrix(features.iloc[: , 0:5], alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# separating the data into training set, validation set, and test set
from sklearn.model_selection import train_test_split

trainDataX, testDataX,  trainDataY, testDataY = train_test_split( features,
                                                                  response,
                                                                  test_size=0.1,
                                                                  random_state=1)

modelDataX, validDataX,  modelDataY, validDataY = train_test_split( trainDataX,
                                                                    trainDataY,
                                                                    test_size=0.25,
                                                                    random_state=1)

trainDataX.shape
testDataX.shape
modelDataX.shape
validDataX.shape

# applying SVM to the data
from sklearn.svm import SVC
clf = SVC(kernel='linear') # possible parameter: kernel = (linear , poly , rbf , sigmoid , precomputed)
clf.fit ( modelDataX , modelDataY )
pred = clf.predict(validDataX)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(validDataY, pred)
# accuracy is 99.87%, but due to the imbalance of the number of actual fruad, we need to test an F1 score
# in order to calculate out of all actual fraud, how many did this model detected.