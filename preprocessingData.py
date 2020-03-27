import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])

# Binarization - used when you want to convert numerical values into boolean values - user presets threshold in sklearn function
# values above, but not equal to 2.1 will be 1 and all values below and equal to 2.1 will be zero
data_binary = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print('Binary data output:\n', data_binary)


# Mena removal - remove the mean from feature vector, each factor is centered on zero - remove bias from the features in feature vector
# first find the mean and standard deviation
mean = input_data.mean(axis=0)
std = input_data.std(axis=0)

# then you remove the mean using built in sklearn preprocessing function
data_scaled = preprocessing.scale(input_data)
scaled_mean = data_scaled.mean(axis=0)
scaled_std = data_scaled.std(axis=0)

# output the differences
print('\nBefore')
print('mean: ', mean ,
      'STD: ', std)
print('\nAfter')
print('mean: ', scaled_mean,
      'STD: ', scaled_std)

# scaling - create a level playing field for the ML algorithm to train on - dont want any feature to be artificially large or small
# min max scaling
data_scalar_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scalar_minmax.fit_transform(input_data)

# the output is scaled so that the max value is 1 and all other values are relative to this value
print("\nMin Max scaled data: \n", data_scaled_minmax)

"""
Normalization - modify values in te feature vector so that we can measure them on a common scale - normally they sump up to 1
l1 normalization - least absolute deviations - make sure that the sum of absolute values is 1 in each row of the feature vector  
l2 normalization - refers to least squares - works by making sure that the sum of squares is 1 
"""
# normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print('Normalized l1: \n', data_normalized_l1)
print('Normalized l2: \n', data_normalized_l2)
