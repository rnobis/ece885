'''GRU Facebook Comments
Trains a GRU RNN on a dataset of Facebook metadata in order to determine 
the number of comments a particular posting will get. This version tries 
to guess the exact number of posts. These were the scores, accuracies, 
and MAE for each data set:

'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.layers import GRU

batch_size = 1 
nb_classes = 10
nb_epoch = 3
score = np.zeros(shape=(5,3))

#Run RNN on each training dataset
for j in range (0, 5):
    print('\nTrain on Dataset {}'.format(j+1))
    #import training data
    if j == 0:
        trainData = np.genfromtxt('Features_Variant_1.csv', delimiter = ",")
    elif j == 1:
        trainData = np.genfromtxt('Features_Variant_2.csv', delimiter = ",")
    elif j == 2:
        trainData = np.genfromtxt('Features_Variant_3.csv', delimiter = ",")
    elif j == 3:
        trainData = np.genfromtxt('Features_Variant_4.csv', delimiter = ",")
    elif j == 4:
        trainData = np.genfromtxt('Features_Variant_5.csv', delimiter = ",")
        
    X_train = trainData[:,0:53]
    Y_train = trainData[:,53]

    #randomly select test data to import
    rtd = np.random.randint(1, 11)

    if rtd == 1:
        testData = np.genfromtxt('Test_Case_1.csv', delimiter = ",")
    elif rtd == 2:
        testData = np.genfromtxt('Test_Case_2.csv', delimiter = ",")
    elif rtd == 3:
        testData = np.genfromtxt('Test_Case_3.csv', delimiter = ",")
    elif rtd == 4:
        testData = np.genfromtxt('Test_Case_4.csv', delimiter = ",")
    elif rtd == 5:
        testData = np.genfromtxt('Test_Case_5.csv', delimiter = ",")
    elif rtd == 6:
        testData = np.genfromtxt('Test_Case_6.csv', delimiter = ",")
    elif rtd == 7:
        testData = np.genfromtxt('Test_Case_7.csv', delimiter = ",")
    elif rtd == 8:
        testData = np.genfromtxt('Test_Case_8.csv', delimiter = ",")
    elif rtd == 9:
        testData = np.genfromtxt('Test_Case_9.csv', delimiter = ",")       
    elif rtd == 10:
        testData = np.genfromtxt('Test_Case_10.csv', delimiter = ",")
    
    X_test = testData[:,0:53]
    Y_test = testData[:,53]

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    #Reshape data to 3 dimensions (time step = 1)
    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
                                            
    #Build neural network model. 
    model = Sequential()
    model.add(GRU(20, input_shape=(1,53), consume_less='mem'))
    model.add(GRU(10))
    model.add(Dense(1))
    
    model.summary()

    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=['accuracy', 'mae'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    score[j] = model.evaluate(X_test, Y_test, verbose=0)

for j in range (0, 5):
    print('Test Score {}: {}'.format(j + 1, score[j,0]))
    print('Test Accuracy {}: {}'.format(j + 1,score[j,1]))
    print('MAE {}: {}'.format(j + 1,score[j,2]))
