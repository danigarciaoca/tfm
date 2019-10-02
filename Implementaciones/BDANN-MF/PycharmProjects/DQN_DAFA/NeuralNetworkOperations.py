from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np


class MyNeuralNetwork(Sequential):
    def __init__(self):
        # instantiate model
        Sequential.__init__(self)

        # input layer
        self.add(Dense(64, input_dim=1, kernel_initializer='uniform'))
        self.add(Dropout(rate=0.5))
        self.add(BatchNormalization())
        self.add(Activation('relu'))

        # # hidden layer
        self.add(Dense(64, kernel_initializer='uniform'))
        self.add(Dropout(rate=0.5))
        self.add(BatchNormalization())
        self.add(Activation('relu'))

        # output layer
        self.add(Dense(1, kernel_initializer='uniform', activation='linear'))

        self.compile(loss='mse', optimizer='adam', metrics=['mae', 'accuracy'])

    def init_NN_params(self):
        # If kernel_initializer='uniform', minval and maxval used by keras are -0.05 and 0.05 respectively
        minval = -0.05
        maxval = 0.05
        layers = self.layers
        for layer in layers:
            if len(layer.weights) != 0:
                if (layer.output_shape == layer.input_shape) and (len(layer.weights) == 4):  # in case of BatchNormalization layer, always 4 "weight" parameters
                    dim = layer.input_shape[1]
                    gamma_init = np.ones(dim)
                    beta_init = np.zeros(dim)
                    moving_mean_init = np.zeros(dim)
                    moving_variance_init = np.ones(dim)
                    w_list = (gamma_init, beta_init, moving_mean_init, moving_variance_init)  # list with total weight parameters (according to BatchNormalization doc)
                else:  # otherwise (i.e, a normal layer sucha as Dense)
                    in_dim = layer.input_shape[1]  # input dimension
                    out_dim = layer.output_shape[1]  # output dimension
                    link_w = np.random.uniform(minval, maxval, (in_dim, out_dim))  # weight term
                    b = np.zeros(out_dim)  # bias term
                    w_list = (link_w, b)  # list with total weight parameters (bias 'b' and weight 'w')
                layer.set_weights(w_list)
