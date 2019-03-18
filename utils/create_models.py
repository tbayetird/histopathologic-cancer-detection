# import the necessary packages
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import SeparableConv2D, Conv2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def create_mlp(shape, depth = 1):

	model = Sequential()

	channelDimension = -1
	initialFilters = 32
	kernelSize = (3, 3)
	convDropout = 0.25
	lastDropout = 0.5
	poolSize = (2, 2)
	k = 2

	model.add(Conv2D(initialFilters, kernelSize, padding="same", input_shape=shape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=channelDimension))
	model.add(MaxPooling2D(pool_size=poolSize))
	model.add(Dropout(convDropout))

	if depth > 1:
		for x in range(2, depth+1):
			filters = initialFilters * k
			for y in range(0, x):
				model.add(Conv2D(filters, kernelSize, padding="same"))
				model.add(Activation("relu"))
				model.add(BatchNormalization(axis=channelDimension))
			model.add(MaxPooling2D(pool_size=poolSize))
			model.add(Dropout(convDropout))
			k *= 2

	model.add(Flatten())
	model.add(Dense(initialFilters * k))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(lastDropout))
	model.add(Dense(1, activation="sigmoid"))

	return model
