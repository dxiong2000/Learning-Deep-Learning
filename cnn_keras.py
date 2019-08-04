import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import History
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

# hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 100

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape data to be usable
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# convert labels to one hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# initialize model
model = Sequential()

# add layers
# first convolutional layer, 32 filters with 5x5 kernel size, ReLU activation function
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation="relu", input_shape=(28, 28, 1)))
# first max pool, 2x2 pool size
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# second convolutional layer, 64 filters with 5x5 kernel size, ReLU activation function
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), activation="relu"))
# second max pool, 2x2 pool size
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
# flatten output as input for fully connected layers
model.add(Flatten())
# hidden layer, 1000 nodes, ReLU activation function
model.add(Dense(units=1000, activation="relu"))
# output layer, 10 nodes, softmax activation function
model.add(Dense(units=10, activation="softmax"))

# compile the model with cost function and optimizer
model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=LEARNING_RATE), metrics=["accuracy"])

# training the model
history = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# test accuracy of the model
final_accuracy = model.evaluate(x=x_test, y=y_test)
print('Test loss:', final_accuracy[0], '|| Test accuracy:', final_accuracy[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
