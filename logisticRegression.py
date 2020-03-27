"""
Technique used to explain the relationship between input variables and output variables - input variables are assume to be independent
and the output variable is referred to as the dependent variable
logistic function is a sigmoid curve - used to build the function with various parameters
building a classifier using logistic regression
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from utilities import visualize_classifier

# define sample input data
X = np.array([[3.1, 7.2], [4, 6.7],
              [2.9, 8], [5.1, 4.5],
              [6, 5], [5.6, 6],
              [3.3, 0.4], [3.9, 0.9],
              [2.8, 1], [0.5, 3.4],
              [1, 4],[0.6, 4.9]])

y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
# create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)
# train the classifier
classifier.fit(X, y)

# visualize performance
visualize_classifier(classifier, X, y)
