# Imports
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

from matplotlib import pyplot as plt

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
tf.random.set_seed(1)

import wandb
print('IMPORTED???')
wandb.init(project="4g4-kim-test-1")
print('Here')

def set_hyperparameter_defaults():

	# hyperparameter_defaults = dict(
	#     num_per_layer = 90,
	#     num_hidden_layers = 4,
	#     learning_rate = 0.001,
	#     momentum = 0.95,
	#     epochs = 1000,
	#     )
	wandb.init(project="4g4-kim-test") #config=hyperparameter_defaults,
	# config = wandb.config
	# print(config)


def get_data():
	print('get_data')
	iris = load_iris()
	X = iris['data']; y = iris['target']
	Y = enc.fit_transform(y[:, np.newaxis]).toarray()
	X_scaled = scaler.fit_transform(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

	return X_train, X_test, Y_train, Y_test


def build_model():

  model = keras.Sequential()
  model.add(layers.Dense(config.num_per_layer, activation='relu', input_dim = mu.shape[1]))
  for l in range(config.num_hidden_layers):
    model.add(layers.Dense(config.num_per_layer, activation='relu'))
  model.add(layers.Dense(output_concat.shape[0]))

  # optimizer = tf.keras.optimizers.RMSprop(config.learning_rate)

  # optimizer = tf.keras.optimizers.RMSprop(0.001)
  optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=config.momentum, nesterov=True)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


def train_swarm():

	my_swarm = NN_Swarm()
	my_swarm.n_particles = 50
	my_swarm.provide_model(model)
	my_swarm.provide_data(X_train, X_test, Y_train, Y_test)
	my_swarm.train(num_epochs=10)
	print("Test set accuracy: {:.3%}".format(my_swarm.get_accuracy()))

if __name__ == "__main__":

	# set_hyperparameter_defaults()
	get_data()
	build_model()
	train_swarm()