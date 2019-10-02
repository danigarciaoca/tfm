import numpy as np
from cartesian import cartesian
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
#plt.ion()

x = np.arange(-50,51)
x_train = cartesian((x,x))
y_train = np.cos(x_train[:, 0]) ** 2 + np.sin(x_train[:, 0] * x_train[:, 1])

model = Sequential()
model.add(Dense(2, activation='relu', input_dim=2))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 1
epochs = 20

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

prediction = model.predict(x_train)

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# W1,b1,W2,b2,W3,b3 = model.get_weights()
#
# import graphviz
#
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')

import numpy as np
from cartesian import cartesian
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import matplotlib.pyplot as plt
#plt.ion()

x = np.arange(-50,51)
x_train = cartesian((x,x))

#y_train = np.cos(x_train[:, 0]) ** 2 + np.sin(x_train[:, 0] * x_train[:, 1])
y_train = np.cos(x)

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=1))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 5
epochs = 20

history = model.fit(x, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

prediction = model.predict(x)

score = model.evaluate(x, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
