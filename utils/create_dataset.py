import numpy as np
import argparse
import cv2
import os
import shutil as sh
import pandas as pd
from setup_dataset import setup

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", type=str, required=False, default='dataset',
	help="path to input dataset")
ap.add_argument("-o", "--output_path", type=str, required=False, default='.',
	help="path to the output dataset")
ap.add_argument("-csv", "--csv", type=str,required=False, default='train_labels.csv',
    help="name of the csv containing informations on the dataset")
ap.add_argument("-n", "--name", type=str, required=False, default='new_dataset',
	help="name of the directory that will be created and filled with the data")
ap.add_argument("-trs", '--train_size', type=int, required=True,
    help ="size of the training dataset")
ap.add_argument("-tes", '--test_size', type=int, required=True,
    help ="size of the test dataset")
args = vars(ap.parse_args())


### get all the files we'll transfer in our new set
ids_path=args['input_path']
csv_path = os.path.join(ids_path,args["csv"])
df_train = pd.read_csv(csv_path)
df_train_trunc= df_train.sample(args["train_size"])

train_files = df_train_trunc['id']
# train_files += ".tif"

test_files=[]
for root,dirs,files in os.walk(os.path.join(ids_path,'test')):
    for file in files:
        if(len(test_files)<args['test_size']):
            test_files.append(file)


### create the new set
new_dataset = os.path.join(args["output_path"],args["name"])
os.system("mkdir {}".format(new_dataset))
os.system("mkdir {}".format(os.path.join(new_dataset,'train')))
os.system("mkdir {}".format(os.path.join(new_dataset,'test')))
df_train_trunc.to_csv(os.path.join(new_dataset,'train_labels.csv'))

for i in train_files:
    sh.copyfile(os.path.join(ids_path,'train',i+'.tif'),
            os.path.join(new_dataset,'train',i+'.tif'))

for i in test_files:
    sh.copyfile(os.path.join(ids_path,'test',i),
            os.path.join(new_dataset,'test',i))


### Restructuring the set trhough setup_dataset :
csv_path = os.path.join(new_dataset,'train_labels.csv')
setup(csv_path,
        args["train_size"],
        os.path.sep.join([new_dataset,'train']),
        os.path.sep.join([new_dataset,'validation']),
        os.path.join(new_dataset,'train')
        )
