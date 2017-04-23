'''ECE885 Final Project
Trains a GRU RNN variant (no input signal, no bias) on a dataset of 
Facebook metadata in order to determine the number of comments a particular
posting will get.


'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from recurrent_v import GRU2
from keras.utils import np_utils

batch_size = 1 #128
nb_classes = 10
nb_epoch = 100

#import training data
trainData = np.genfromtxt('Features_Variant_1.csv', delimiter = ",")
#trainData = np.genfromtxt('Features_Variant_2.csv', delimiter = ",")
#trainData = np.genfromtxt('Features_Variant_3.csv', delimiter = ",")
#trainData = np.genfromtxt('Features_Variant_4.csv', delimiter = ",")
#trainData = np.genfromtxt('Features_Variant_5.csv', delimiter = ",")
X_train = trainData[:,0:53]
y_train = trainData[:,53]

#import test data
testData = np.genfromtxt('Test_Case_1.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_1.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_2.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_3.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_4.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_5.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_6.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_7.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_8.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_9.csv', delimiter = ",")
#testData = np.genfromtxt('Test_Case_10.csv', delimiter = ",")
X_test = testData[:,0:53]
y_test = testData[:,53]

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#Reclassify Output Data to range of comments
i = 0
for i in range (0, y_train.shape[0]):
    if y_train[i] == 0:
        y_train[i] = 1
    elif y_train[i] > 0 and y_train[i] < 5:
        y_train[i] = 2
    elif y_train[i] >= 5 and y_train[i] < 10:
        y_train[i] = 3
    elif y_train[i] >= 10 and y_train[i] < 50:
        y_train[i] = 4
    elif y_train[i] >= 50 and y_train[i] < 100:
        y_train[i] = 5
    elif y_train[i] >= 100 and y_train[i] < 500:
        y_train[i] = 6
    elif y_train[i] >= 500 and y_train[i] < 1000:
        y_train[i] = 7
    elif y_train[i] >= 1000 and y_train[i] < 5000:
        y_train[i] = 8
    elif y_train[i] >= 5000 and y_train[i] < 10000:
        y_train[i] = 9
    elif y_train[i] >= 10000:
        y_train[i] = 10
               
for i in range (0, y_test.shape[0]):
    if y_test[i] == 0:
        y_test[i] = 1
    elif y_test[i] > 0 and y_test[i] < 5:
        y_test[i] = 2
    elif y_test[i] >= 5 and y_test[i] < 10:
        y_test[i] = 3
    elif y_test[i] >= 10 and y_test[i] < 50:
        y_test[i] = 4
    elif y_test[i] >= 50 and y_test[i] < 100:
        y_test[i] = 5
    elif y_test[i] >= 100 and y_test[i] < 500:
        y_test[i] = 6
    elif y_test[i] >= 500 and y_test[i] < 1000:
        y_test[i] = 7
    elif y_test[i] >= 1000 and y_test[i] < 5000:
        y_test[i] = 8
    elif y_test[i] >= 5000 and y_test[i] < 10000:
        y_test[i] = 9
    elif y_test[i] >= 10000:
        y_test[i] = 10

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#Reshape data to 3 dimensions (time step = 1)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
                                            
#Build neural network model. 
model = Sequential()
model.add(GRU2(25, input_shape=(1,53), consume_less='mem'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=0.1, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])