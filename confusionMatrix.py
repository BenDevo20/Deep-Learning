"""
figure or table that is used to describe the performance of a classifier. Extracted from a test dataset for which the ground truth is know
compare each class with every other class and see how many samples are misclassified
type 1 error - predicted value is 1, but true value is 0 - false pos
type 2 error - predicted value is 0, but true value is 1 - false neg
"""

# confusion matrix example for biometric data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# sample labels
true_lab = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_lab = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# confusion matrix
confusion_mat = confusion_matrix(true_lab, pred_lab)

# visualizing the data
# white indicated higher values - black lower values - want diagonals to be all white -- 100% accuracy
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True Value')
plt.xlabel('Predicted Values')
plt.show()

# print the classification report
targets = ['class-0', 'class-1', 'class-2', 'class-3', 'class-4']
print('\n', classification_report(true_lab, pred_lab, target_names=targets))

