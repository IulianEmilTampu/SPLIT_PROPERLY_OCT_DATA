# %%
'''
Script that, given the folder where the detaset is located, prints out examples
of testing and training datasets as they result from the data loader
'''

import os
import sys
import cv2
import glob
import json
import pickle
import random
import pathlib
# import imutils
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import to_categorical
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# custom imports
import utilities
import utilities_models_tf

# %% load the data that the model used for training and testing
# get dataset info from the configuration file
from_configuration_file = True

if from_configuration_file:
    # model_name = 'LightOCT_per_image_split_5_folds_rkf_2_lr0.0001_batch_64_Srinivas_rls_True'
    # model_name = 'LightOCT_per_volume_split_5_folds_rkf_10_lr0.0001_batch64_AIIMS_rls_False'
    model_name = 'LightOCT_per_image_split_3_folds_lr0.0001_batch64_retinal'
    trained_models_path = '/flush/iulta54/Research/P3_OCT_SPLIT_PROPERLY_YOUR_DATA/trained_models/Pre-revision'
    # dataset_path = '/flush/iulta54/Research/Data/OCT/Retinal/Srinivas/'
    # dataset_path = '/flush/iulta54/Research/Data/OCT/AIIMS_Dataset/original'
    dataset_path = '/flush/iulta54/Research/Data/OCT/Retinal/Zhang_dataset/per_class_files'

    # load configuration file
    with open(os.path.join(trained_models_path, model_name,'config.json')) as json_file:
        config = json.load(json_file)

    # take one testing. training and validation images (tr and val for fold specific fold)
    # make sure that the files point to this system dataset
    fold = 0
    # test_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1])  for f in config['test']]
    # random.shuffle(test_img)
    # tr_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1])  for f in config['training'][fold]]
    # val_img = [os.path.join(dataset_path, pathlib.Path(f).parts[-2], pathlib.Path(f).parts[-1])  for f in config['validation'][fold]]

    # fix names based on the given dataset path
    if any([config['dataset_type'] == 'retinal', config['dataset_type'] == 'Kermany']):
        idx =  3
    elif config['dataset_type'] == 'AIIMS':
        idx =  4
    elif config['dataset_type'] == 'Srinivas':
        idx =  5
    
    test_img, tr_img, val_img = [], [], []
    for set_name, set in zip(('test', 'training', 'validation'),(test_img, tr_img, val_img)):
        # get paths for this set
        if set_name == 'test':
            file_names = config[set_name]
        else:
            file_names = config[set_name][fold]
        # build file names to point to this given dataset
        for f in file_names:
            aus = [pathlib.Path(f).parts[-i] for i in reversed(range(idx))][0:-1]
            aus.insert(0, dataset_path)
            set.append(os.path.join(*aus))
    
    # some other settings
    crop_size = config['input_size'] # (h, w)
else:
    # specify manually the files to show
    test_img = []
    tr_img = []
    val_img = []
    crop_size = []

    # dataset_path = '/home/iulta54/Desktop/Testing/TH_DL_dummy_dataset/Created/LigthOCT_TEST_isotropic_20000s'
    # file_names = glob.glob(os.path.join(dataset_path, '*'))
    # c_type='c1'
    # file_names, labels, organized_files = utilities.get_organized_files(file_names, c_type, categorical=False)

examples_to_show = 80

# %% 2 create dataset and augmentation layer
importlib.reload(utilities)

# create generator based on model specifications and dataset
if any([config['dataset_type'] == 'retinal', config['dataset_type'] == 'Kermany']):
    data_gen = utilities.Kermany_data_gen
elif config['dataset_type'] == 'AIIMS':
    data_gen = utilities.AIIMS_data_gen
elif config['dataset_type'] == 'Srinivas':
    data_gen = utilities.Srinivas_data_gen

dataset =  data_gen(tr_img,
                        unique_labels=config['unique_labels'],
                        batch_size=16,
                        training=False,
                        channels=config['n_channels'],
                        input_size=config['input_size'],
                        normalize=True)

augmentor = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.02)],
        name='Augmentation')


# %% 2 create data augmentation layers
importlib.reload(utilities)

x, y = next(iter(dataset))

# x = augmentor(x, training=True)
sample = (x.numpy(), y.numpy())

print(f'{"Mean:":5s}{x.numpy().mean():0.2f}')
print(f'{"STD:":5s}{x.numpy().std():0.2f}')

utilities.show_batch_2D(sample, img_per_row=5)

# %% 3 show images along with their histogram
importlib.reload(utilities)
utilities.show_batch_2D_with_histogram(sample)


# %%
