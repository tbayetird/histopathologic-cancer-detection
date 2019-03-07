#%%
from config import config
from imutils import paths
import os
import utils.setup_dataset
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as backend
import imp
from utils import create_models as cm

################################
# SETTING UP DATASET
###############################

shouldSetup = False
if (shouldSetup):
    utils.setup_dataset.setup()

batch_size = 16

# Set the image size.
img_height = 96
img_width = 96

################################
# LOADING AND PREPROCESSING THE DATA
################################

train_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                                   rescale=1./255,
                                   width_shift_range=4,
                                   height_shift_range=4,
                                   rotation_range=90,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x, rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="binary",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

valid_generator = test_datagen.flow_from_directory(
    config.VAL_PATH,
    class_mode="binary",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

#%%
################################
# LOADING MODEL
################################

imp.reload(utils.create_models)
backend.clear_session()
model = cm.create_mlp((img_width,img_height,3),True)
model.summary()

#%%
################################
# TRAINING MODEL
################################

model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-5),
            metrics=['acc'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        epochs=10,
        validation_data=valid_generator,
        validation_steps=3)


################################
# TRAINING VISUALS
###############################

# Nothing yet


################################
# PREDICTING ON TEST SET
###############################


#Nothing yet
