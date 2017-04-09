from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import os.path
#
#Models of the Circuits, Systems and Neural Networks (CSANN) Lab--MSU
#
pth = "C:\Users\salem\LAYERS\NEW1"
os.chdir(pth)
from Variants import LSTM3
#from keras.layers import LSTM


np.random.seed(3)
srgn = RandomStreams(3)

batch_size = 32
nb_classes = 10
nb_epochs = 10
hidden_units = 100


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



np.random.seed(3)
srgn = RandomStreams(3)


act = 'tanh'
lstm = LSTM3
eta = 2e-3

model = Sequential()

lstm3 = lstm(consume_less='mem',output_dim=hidden_units,
                    activation=act,
                    input_shape=X_train.shape[1:])

model.add(lstm3)
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=eta, rho=0.9, epsilon=1e-8, decay=0)
model.compile(loss='categorical_crossentropy',
              optimizer= 'rmsprop',
              metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                 verbose=1, validation_data=(X_test, Y_test))