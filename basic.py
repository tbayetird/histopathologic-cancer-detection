import os

import utils.create_models
local_dir = os.path.abspath('.')
mock_data_dir = os.path.join(local_dir,'mock_data')
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
# PREPROCESSING DATAS
###############################


#Nothing yet


################################
# TRAINING MODEL
###############################


#Nothing yet


################################
# TRAINING VISUALS
###############################

# Nothing yet


################################
# PREDICTING ON TEST SET
###############################


#Nothing yet
