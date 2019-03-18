import os

ROOT_FOLDER=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATASET_FOLDER = os.path.join(ROOT_FOLDER,"new_dataset")
ORIG_INPUT_TRAINING_DATASET =os.path.join(DATASET_FOLDER,'train')
ORIG_INPUT_TESTING_DATASET =os.path.join(DATASET_FOLDER,'test')
BASE_PATH = DATASET_FOLDER
TRAIN_PATH = os.path.sep.join([BASE_PATH, "train"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])
SUBMISSION_TEST_PATH = os.path.sep.join([ROOT_FOLDER, "dataset", "test"])
LABELS_PATH = os.path.sep.join([BASE_PATH, "train_labels.csv"])
SUBMITION_FILE_PATH = BASE_PATH
DATASET_SIZE = 10000
VAL_SPLIT = 0.2
MODELS_PATH = os.path.join(ROOT_FOLDER,"models")
