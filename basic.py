from config import config
from imutils import paths
import os
import utils.create_models
import utils.setup_dataset
from keras.preprocessing.image import ImageDataGenerator

################################
# SETTING UP DATASET
###############################
 
shouldSetup = True
if (shouldSetup): 
    utils.setup_dataset.setup()

batch_size = 32

# Set the image size.
img_height = 96
img_width = 96

################################
# LOADING MODEL
###############################

model = utils.create_models.create_mlp((img_width,img_height,3))
model.summary()

################################
# LOADING AND PREPROCESSING THE DATA
###############################

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
    class_mode="categorical",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

valid_generator = test_datagen.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)


################################
# TRAINING MODEL
###############################




################################
# TRAINING VISUALS
###############################

# Nothing yet


################################
# PREDICTING ON TEST SET
###############################


#Nothing yet
