import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

# seed number generator for repeatable results
np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# if condition is not met - code will stop and output error message
assert(X_train.shape[0] == y_train.shape[0], "the number of images is not equal to the number of labels")
assert(X_test.shape[0] == y_test.shape[0], "the number of images is not equal to the number of labels")
assert(X_train.shape[1:] == (28,28)), "the dimensions of the images are not 28x28"
assert(X_test.shape[1:] == (28,28)), "the dimensions of the images are not 28x28"

num_of_samples = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(cols, 10))
fig.tight_layout()

for i in range(cols):
    for j in range(num_classes):
        # split training and test set
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis('off')
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title('Distribution of the training dataset')
plt.xlabel('class number')
plt.ylabel('number of images')
#plt.show()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# normalization process - max pixel value is 255 - all output is between 0 and 1
X_train = X_train/255
X_test = X_test/255

num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

# create neural net
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())
# splitting datase
model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=200, verbose=1, shuffle=1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epochs')


plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['acc', 'val_acc'])
plt.title('accuracy')
plt.xlabel('epochs')

score = model.evaluate(X_test, y_test, verbose=0)
print(score)

import requests
from PIL import Image

url = 'https://colah.github.io/posts/2014-10-Visualizing-MNIST/img/mnist_pca/MNIST-p1815-4.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)

import cv2
img_array = np.asarray(img)
resized = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)
image = image/255
image = image.reshape(1,784)

prediction = model.predict_classes(image)


