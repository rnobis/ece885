'''ECE885 Mini-Project 2 Part 3
Trains a LSTM RNN on the MNIST dataset.

Updated MP2_Part2_RNobis.py to have a variable learning rate based on the
loss function, to see the results.
'''

from __future__ import print_function
import numpy as np
import keras.callbacks as cb
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler

batch_size = 32 #128
nb_classes = 10
nb_epoch = 50

sd=[]
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        #Create array for losses
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        #Populate arrays for losses and step decays, print out learning rate
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('\nlearning rate:', step_decay(len(self.losses)))

def step_decay(losses):
    #Calculate learning rate based on loss function
    lrate=0.039894*np.exp(np.array(temp_history.losses[-1]))
    return lrate

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Build neural network model. 
model = Sequential()
model.add(LSTM(50, input_shape=(28,28)))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#Callbacks to run after each epoch
temp_history = LossHistory()
lr = LearningRateScheduler(step_decay)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test), callbacks=[lr, temp_history])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])