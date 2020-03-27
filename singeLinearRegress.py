import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# input tet file from directory - example data
input_file = 'data_singlevar_regr.txt'

# reading the data
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# splitting into train and test data
num_train = int(0.8*len(X))
num_test = len(X) - num_train
# train
X_train, y_train = X[:num_train], y[:num_train]
# test
X_test, y_test = X[num_train:], y[num_train:]

# create linear regression object
regress = linear_model.LinearRegression()
regress.fit(X_train, y_train)
# predict the output
y_test_pred = regress.predict(X_test)

# plotting the results and output
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Compute performance metrics
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# pickle the model
output_file = 'model.pkl'
# saving the model
with open(output_file, 'wb') as f:
    pickle.dump(regress, f)


"""
This is the code used to load the model 

with open(output_model_file, 'rb') as f:
    regress_model = pickle.load(f)
"""