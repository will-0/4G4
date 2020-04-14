# Imports

import wandb

hyperparameter_defaults = dict(
    num_per_layer = 10,
    num_hidden_layers = 1,
    learning_rate = 0.001,
    epochs = 100,
 	num_particles = 50,
 	c_1 = 2,
 	c_2 = 2,
 	v_max = 0.1
    )
wandb.init(config=hyperparameter_defaults, project="4g4-kim-test")
config = wandb.config
print(config)


import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

# from matplotlib import pyplot as plt

import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from IPython.display import SVG
from keras.utils import model_to_dot
import pydot

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
enc = OneHotEncoder(); scaler = StandardScaler();

from SwarmParty import NN_Swarm

# Set seed for reproducible results
np.random.seed(1)
tf.compat.v1.set_random_seed(1)


def get_data():
	iris = load_iris()
	X = iris['data']; y = iris['target']
	Y = enc.fit_transform(y[:, np.newaxis]).toarray()
	X_scaled = scaler.fit_transform(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.4, random_state=2)

	return X_train, X_test, Y_train, Y_test


def build_model():

	model = keras.Sequential()
	model.add(layers.Dense(config.num_per_layer, activation='relu', input_shape=(4,)))
	for l in range(config.num_hidden_layers):
		model.add(layers.Dense(config.num_per_layer, activation='relu'))
	model.add(layers.Dense(3))

	optimizer = tf.keras.optimizers.RMSprop(config.learning_rate)

	model.compile(loss='mse',
	        optimizer=optimizer)
	return model


def train_swarm():
	wandb.init(project="4g4-kim-test-1")
	my_swarm = NN_Swarm(n_particles = config.num_particles, x_max = 1, v_max = config.v_max, c_1 = config.c_1, c_2 = config.c_2)
	my_swarm.provide_model(model)
	my_swarm.provide_data(X_train, X_test, Y_train, Y_test)
	my_swarm.train(num_epochs=config.epochs)

	print("Test set accuracy: {}".format(str(my_swarm.get_accuracy())))

if __name__ == "__main__":

	X_train, X_test, Y_train, Y_test = get_data()
	model = build_model()
	model.summary()
	train_swarm()