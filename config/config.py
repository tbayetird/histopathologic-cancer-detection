import os

ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATASET_FOLDER = os.path.join(ROOT_FOLDER,"mock-data")
ORIG_INPUT_TRAINING_DATASET = DATASET_FOLDER + "/train"
ORIG_INPUT_TESTING_DATASET = DATASET_FOLDER + "/test"
BASE_PATH = DATASET_FOLDER
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])
LABELS_PATH = os.path.sep.join([BASE_PATH, "mock_train_labels.csv"])
DATASET_SIZE = 15
VAL_SPLIT = 0.2
MODEL_PATH='' # no model to load yet
