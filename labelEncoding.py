import numpy as np
from sklearn import preprocessing


# convert word labels into numerical representations is label encoding - enables the algorithms to operate on data
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# print the mapping between words and numbers
print("\nLabel mapping: ")
for i, item in enumerate(encoder.classes_):
    print(item, '--->', i)

# encode a set of random ordered labels to see how it performs
test_labels = ['green', 'red', 'black']
encoded_labels = encoder.transform(test_labels)
print('\nLabels =', test_labels)
print('Encoded values =', list(encoded_labels))

# decode a random set of numbers
encoded_labels = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_labels)
print('\nEncoded Values =', encoded_labels)
print('\nDecoded labels =', list(decoded_list))

