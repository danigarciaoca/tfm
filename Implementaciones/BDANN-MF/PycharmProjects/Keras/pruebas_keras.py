from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from cartesian import cartesian
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization

import gym

env = gym.make('CartPole-v0')
# #Red data from csv file for training and validation data
# TrainingSet = np.genfromtxt("training.csv", delimiter=",", skip_header=True)
# ValidationSet = np.genfromtxt("validation.csv", delimiter=",", skip_header=True)
#
# # split into input (X) and output (Y) variables
# X1 = TrainingSet[:,0:5]
# Y1 = TrainingSet[:,5]
# x_train = X1
# y_train = Y1
#
# X2 = ValidationSet[:,0:5]
# Y2 = ValidationSet[:,5]


# x_train = cartesian((x,x))
x_train = 2*np.pi*np.random.rand(50000,2)
y_train = np.cos(x_train[:, 0]) ** 2 + np.sin(x_train[:, 0] * x_train[:, 1])

# instantiate model
model = Sequential()

# input layer
model.add(Dense(512, input_dim=x_train.shape[1], kernel_initializer='uniform'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())
model.add(Activation('relu'))

# # hidden layer
model.add(Dense(512, kernel_initializer='uniform'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output layer
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))

# setting up the optimization of our weights
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer='adam', metrics=['mae','accuracy'])

# running the fitting
model.fit(x_train, y_train, epochs=1000, batch_size=2048, verbose=2)


# model = Sequential()
# model.add(Dense(8, activation='relu', input_dim=1))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='tanh'))
#
# model.summary()
#
# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
#
# batch_size = 32
# epochs = 50
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1)
#
# prediction = model.predict(x_train)
#
# score = model.evaluate(x_train, y_train, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

W1,b1,W2,b2,W3,b3 = model.get_weights()
#
# import graphviz
#
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

# import numpy as np
# from cartesian import cartesian
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD, Adam, RMSprop
# from keras.utils import np_utils
#
# import matplotlib.pyplot as plt
# #plt.ion()
#
# x = np.arange(-50,51)
# x_train = cartesian((x,x))
#
# #y_train = np.cos(x_train[:, 0]) ** 2 + np.sin(x_train[:, 0] * x_train[:, 1])
# y_train = np.cos(x)
#
# model = Sequential()
# model.add(Dense(8, activation='relu', input_dim=1))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='tanh'))
#
# model.summary()
#
# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
#
# batch_size = 5
# epochs = 20
#
# history = model.fit(x, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1)
#
# prediction = model.predict(x)
#
# score = model.evaluate(x, y_train, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
