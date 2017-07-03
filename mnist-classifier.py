# the next two imports are to ignore few tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import idx2numpy
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import regularizers
from keras import metrics
from keras.utils.np_utils import to_categorical
from keras import optimizers
from scipy import misc
# folder url where data is stored
DATA_PATH = 'data/'

# conver idx to np format and load them
X_train = idx2numpy.convert_from_file(DATA_PATH + 'train-images.idx3-ubyte')
Y_train = idx2numpy.convert_from_file(DATA_PATH + 'train-labels.idx1-ubyte')
X_test = idx2numpy.convert_from_file(DATA_PATH + 't10k-images.idx3-ubyte')
Y_test = idx2numpy.convert_from_file(DATA_PATH + 't10k-labels.idx1-ubyte')

misc.imsave('new2.png',X_train[0])

# reshape the data so as to fit the format of (samples, channels, width, height)
X_train = X_train.reshape(60000, 1, 28, 28).astype('float32')
X_test = X_test.reshape(10000, 1, 28, 28).astype('float32')

Y_train = Y_train.reshape(60000)
Y_test = Y_test.reshape(10000)
print(X_train[0])
print(Y_train[0])

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print("\n\n")
print(Y_train[0])


# print("X shape is ", X_test.shape)
# print("Y shape is", Y_test.shape)
#
# print("PEACE!")

# MODEL DEFINITION
model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.02), strides=(1,1), padding='valid', activation='relu', data_format='channels_first', input_shape=(1,28,28)))
model.add(Conv2D(filters=10, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.02), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))

model.add(Conv2D(filters=12, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.02), strides=(1,1), padding='valid', activation='relu'))
model.add(Conv2D(filters=12, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.02), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))

model.add(Conv2D(filters=12, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.02), strides=(1,1), padding='valid', activation='relu'))
model.add(Conv2D(filters=12, kernel_size=(2,2), kernel_regularizer=regularizers.l2(0.02), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))


model.add(Flatten())

model.add(Dense(units=15,activation='sigmoid', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(rate=0.05,seed=7))
model.add(Dense(units=10,activation='sigmoid', kernel_regularizer=regularizers.l2(0.02)))

# MODEL COMPILATION
# reduce the learning rate if training accuracy suddenly drops and keeps decreasing
sgd = optimizers.SGD(lr=0.003) # lr by default is 0.01 for SGD

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy])

# MODEL FIT
model.fit(X_train, Y_train, epochs=12, batch_size=30)
model.save('mnist-classifier-model.h5')
model.save_weights(filepath='mnist-classifier-weights.h5')

# MODEL EVALUATION
print("\nEvaluating the model on test data. This won't take long. Relax!")
pred = model.predict(X_test,batch_size=1)
score  = metrics.categorical_accuracy(Y_test, pred)
print("\nAccuracy on test data : ", score)

# manually end the session to avoid occasional exceptions while running the program
from keras import backend as K
K.clear_session()
