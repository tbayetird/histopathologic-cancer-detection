import argparse
import gc
import os
from glob import glob
from time import gmtime, strftime

import keras
import numpy as np
import pandas as pd
from keras.models import model_from_json
from skimage.io import imread

from config import config

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--model_name", type=str, required=True, help="name of model")
args = vars(ap.parse_args())

model_name = args['model_name']
print("[INFO] Loading model '{}' from disk".format(model_name))

json_file = open(config.MODELS_PATH + "/" + model_name + ".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(config.MODELS_PATH + "/" + model_name + ".h5")

print("[INFO] Loaded model '{}' from disk".format(model_name))

print("[INFO] Predicting and generating submission file")

testingDataset = glob(os.path.join(config.SUBMISSION_TEST_PATH,'*.tif'))
submissionDataframe = pd.DataFrame()
testingDatasetSize = len(testingDataset)
print("[INFO] Predicting on '{}' data".format(testingDatasetSize))

testing_batch_size = 32
for index in range(0, testingDatasetSize, testing_batch_size):
    print("[INFO] Predicting on batch: %i - %i"%(index, index+testing_batch_size))
    df = pd.DataFrame({'path': testingDataset[index:index+testing_batch_size]})
    df['id'] = df.path.map(lambda x: os.path.basename(x).split(".")[0])
    df['image'] = df['path'].map(imread)
    stack = np.stack(df['image'].values)
    stack = keras.applications.inception_resnet_v2.preprocess_input(stack)
    predictions = loaded_model.predict(stack)
    print(predictions[4])
    exit(0)
    df['label'] = predictions
    #submissionDataframe = pd.concat([submissionDataframe, df[['id', 'label']]])

print("[INFO] Generating submission file")

#submissionFilePath = os.path.sep.join([config.SUBMITION_FILE_PATH, ("submission__" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".csv")])
#submissionDataframe.to_csv(submissionFilePath, index = False, header = True)

#clearing ram, make some free space
gc.collect()

print("[INFO] Done")