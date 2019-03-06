import os
import keras
import pandas as pd
import numpy as np
import utils.create_models
from glob import glob

from keras.preprocessing.image import load_img, img_to_array


file_loc = os.path.realpath(__file__)
local_dir = os.path.dirname(file_loc)
mock_data_dir = os.path.join(local_dir,'mock-data')
model_path='' # no model to load yet

# Set the image size.
img_height = 96
img_width = 96

################################
# LOADING MODEL
###############################

model = utils.create_models.create_mlp((96,96,3))
model.summary()

################################
# LOADING AND PREPROCESSING THE DATA
###############################


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
