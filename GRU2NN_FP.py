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

batch_size = 32 #128
nb_classes = 100000
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

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



#Reshape data to 3 dimensions (time step = 1)
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
                                            
#Build neural network model. 
model = Sequential()
model.add(GRU2(100, input_shape=(1,53), consume_less='mem'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr=0.001, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

