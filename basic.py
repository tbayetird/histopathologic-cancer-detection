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
import gc
from glob import glob
import pandas as pd
import shutil
from skimage.io import imread
import numpy as np
from time import gmtime, strftime

################################
# SETTING UP DATASET
###############################
#
# shouldSetup = False
# if (shouldSetup):
#     print("[INFO] Setting up dataset")
#     utils.setup_dataset.setup()
## Previous way of working with datasets, should be deleted


batch_size = 100
testing_batch_size = 5000

# Set the image size.
img_height = 96
img_width = 96

################################
# LOADING AND PREPROCESSING DATA
################################

print("[INFO] Loading and preprocessing data")
print("[INFO] Testing on '{}' data".format(len(os.listdir(config.TEST_PATH))))

train_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                                   rescale=1./255,
                                   width_shift_range=4,
                                   height_shift_range=4,
                                   rotation_range=90,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x, rescale=1./255)

print("[INFO] Loading and preprocessing training data")
train_generator = train_datagen.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="binary",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

print("[INFO] Loading and preprocessing validation data")
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

print("[INFO] Loading model")

imp.reload(utils.create_models)
backend.clear_session()
model = cm.create_mlp((img_width,img_height,3),True)
model.summary()

#%%
################################
# TRAINING MODEL
################################

print("[INFO] Training model")

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

#%%
################################
# PREDICTING ON TEST SET
###############################

print("[INFO] Predicting and generating submission file")

testingDataset = glob(os.path.join(config.TEST_PATH,'*.tif'))
submissionDataframe = pd.DataFrame()
testingDatasetSize = len(testingDataset)
print("[INFO] Predicting on '{}' data".format(testingDatasetSize))

for index in range(0, testingDatasetSize, testing_batch_size):
    print("[INFO] Predicting on batch: %i - %i"%(index, index+testing_batch_size))
    df = pd.DataFrame({'path': testingDataset[index:index+testing_batch_size]})
    df['id'] = df.path.map(lambda x: os.path.basename(x).split(".")[0])
    df['image'] = df['path'].map(imread)
    stack = np.stack(df['image'].values)
    stack = (stack - stack.mean()) / stack.std()
    predictions = model.predict(stack)
    df['label'] = predictions
    submissionDataframe = pd.concat([submissionDataframe, df[['id', 'label']]])

print("[INFO] Generating submission file")

imp.reload(config)
submissionFilePath = os.path.sep.join([config.SUBMITION_FILE_PATH, ("submission__" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".csv")])
submissionDataframe.to_csv(submissionFilePath, index = False, header = True)

#clearing ram, make some free space
gc.collect()

print("[INFO] Done")
