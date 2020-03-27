"""
used to build classifier using Bayes theorem - describes the probability of an event occurring based on different conditions
that are related to the event
problem instances are represented as vectors of feature values - assigning class labels to problem instances
The assumption - the value of any given feature is independent of the value of any other feature - the naive part
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from utilities import visualize_classifier
from sklearn.model_selection import cross_val_score

# inputting file from directory - example file containing test data
inout_file = 'data_multivar_nb.txt'
# load data from test file
data = np.loadtxt(inout_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# create an instance of the Naive bayes classifier - assume they follow gaussian distribution
classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)

# compute the accuracy
accuracy = 100.0*(y==y_pred).sum() / X.shape[0]
print('Accuracy: ', round(accuracy, 2), "%")
visualize_classifier(classifier, X, y)

# need to split the data into training and test subsets - a normal allocation percentage is 80 - train and 20 - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# computing the accuracy of the classifier and visualizing the performance
accuracy = 100.0*(y_test==y_test_pred).sum() / X_test.shape[0]
print('Accuracy: ', round(accuracy, 2), "%")
visualize_classifier(classifier_new, X_test, y_test)

# inbuilt functions to calculate the accuracy, precision and recall values based on threefold cross validation
num_folds = 3
accuracy_values = cross_val_score(classifier,
X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier,
X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier,
X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier,
X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")