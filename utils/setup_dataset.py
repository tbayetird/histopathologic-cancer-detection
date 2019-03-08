from config import config
from imutils import paths
import pandas as pd
import random
import shutil
import os
from sklearn.model_selection import train_test_split

def initialSetup():
    dataframe = pd.read_csv(config.LABELS_PATH)
    i = int(len(dataframe) * config.VAL_SPLIT)

    dataframeSubset = dataframe.sample(n=config.DATASET_SIZE, random_state=2018)
    trainingSet, validationSet = train_test_split(dataframeSubset,test_size=config.VAL_SPLIT)


    trainDatasets = [
        ("training", trainingSet, config.TRAIN_PATH),
        ("validation", validationSet, config.VAL_PATH)
    ]

    print("[INFO] working on '{}' data".format(config.DATASET_SIZE))

    for (dType, dataset, baseOutput) in trainDatasets:
        print("[INFO] building '{}' split".format(dType))

        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)

        for (index, row) in dataset.iterrows():
            label = row['label']
            filename = row['id'] + ".tif"
            labelPath = os.path.sep.join([baseOutput, str(label)])

            if not os.path.exists(labelPath):
                print("[INFO] 'creating {}' directory".format(labelPath))
                os.makedirs(labelPath)

            originalFilePath = os.path.sep.join([config.ORIG_INPUT_TRAINING_DATASET, filename])
            filePath = os.path.sep.join([labelPath, filename])
            shutil.move(originalFilePath, filePath)

        print("[INFO] done building '{}' split".format(dType))

    print("[INFO] all done, good to go")


def setup(csv_path,dataset_size,train_path,val_path,train_dir):
    dataframe = pd.read_csv(csv_path)
    i = int(len(dataframe) * config.VAL_SPLIT)

    dataframeSubset = dataframe.sample(n=dataset_size, random_state=2018)
    trainingSet, validationSet = train_test_split(dataframeSubset,test_size=config.VAL_SPLIT)


    trainDatasets = [
        ("training", trainingSet,train_path),
        ("validation", validationSet, val_path)
    ]

    print("[INFO] working on '{}' data".format(dataset_size))

    for (dType, dataset, baseOutput) in trainDatasets:
        print("[INFO] building '{}' split".format(dType))

        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)

        for (index, row) in dataset.iterrows():
            label = row['label']
            filename = row['id'] + ".tif"
            labelPath = os.path.sep.join([baseOutput, str(label)])

            if not os.path.exists(labelPath):
                print("[INFO] 'creating {}' directory".format(labelPath))
                os.makedirs(labelPath)

            originalFilePath = os.path.sep.join([train_dir, filename])
            filePath = os.path.sep.join([labelPath, filename])
            shutil.move(originalFilePath, filePath)

        print("[INFO] done building '{}' split".format(dType))

    print("[INFO] all done, good to go")
