# import the necessary packages
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def create_mlp(shape, classes):

	# define our MLP network

	model = Sequential()
	channelDimension = -1

	model.add(SeparableConv2D(32, (3, 3), padding="same", input_shape=shape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(SeparableConv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(SeparableConv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(SeparableConv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(SeparableConv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(SeparableConv2D(128, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(classes))
	model.add(Activation("sigmoid"))


	# return our model
	return model
