"""
classifier that is defined using a separating hyperplane between the classes
hyperplane - n-dimensional version of a line
SVM finds the optimal hyperplane that separates the training data into two classes
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# income data example - based on example txt file in directory
input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        # the last eleement in each line represents the label - need to split it accordingly - commas
        data = line[:-1].split(', ')
        if data[-1] == '<=50k' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50k' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# convert list into array - data type can be given as input for sklearn functions
X = np.array(X)

# need to encode labels that are strings - will output multiple label encoders
label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# creating svm classifier with a linear kernel
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X, y)

# cross validation 80/20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# compute f1 score for the classifier
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print('f1 score: ', f1)

# predicting output
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Encode test datapoint
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform(input_data[i]))
        count += 1
input_data_encoded = np.array(input_data_encoded)

# running the classifier
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class)[0])

