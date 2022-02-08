'''
Script that configures the training of a deep learning model for disease
classification. This code is designed to work for the two following datasets:
1 - OCT2017: doi 10.17632/rscbjbr9sj.2
             https://data.mendeley.com/datasets/rscbjbr9sj/2.
2 - AIIMS dataset: https://www.bioailab.org/datasets

The model set by this configuration script is the LigthOCT model proposed by
Butola et al., in 2020. DOI https://doi.org/10.1364/BOE.395487

The configuration script takes inline inputs. For a description of all available
input parameters, runt configure_training.py --help
'''

import os
import sys
import json
import glob
import types
import time
import pathlib
import random
import argparse
import importlib
import numpy as np
from random import shuffle
from datetime import datetime
import matplotlib.pyplot as plt
from operator import itemgetter
from shutil import copyfile, move
from collections import OrderedDict
from sklearn.model_selection import KFold

## define utilities
def get_retinal_organized_files(dataset_folder):
    '''
    Utility that given the path to the OCT2017 containing the folders of
    the different classes, organises the files in a dictionary nested as:
    class_name
        subject_ID or volume ID
            file name

    INPUT
    dataset_folder : str
            Path to the folder containing the folder of the images organized per
            class. The organization of the files as per requirement can be obtained
            by running the refine_dataset.py script on original version of the
            dataset.

    OUTPUT
    organized_files : dict
            A dictionary where the data is organized in hierarchical fashion:
            class_name
                subject_ID or volume ID
                    file name
    '''
    # get class names and create dictionary
    classes_folders = glob.glob(os.path.join(dataset_folder, '*/'))

    organized_files = dict.fromkeys([os.path.basename(os.path.dirname(f)) for f in classes_folders])

    # loop through the classes, get all the unique subject ID or volumes IDs
    for (idx, cls), class_folder in zip(enumerate(organized_files.keys()), classes_folders):
        class_files = glob.glob(os.path.join(class_folder,'*'))
        unique_ids = list(dict.fromkeys([f[f.find('-')+1:f.rfind('-')] for f in class_files]))
        organized_files[cls] = dict.fromkeys(unique_ids)
        # get all the files for all the unique ids
        for idy, ids in enumerate(unique_ids):
            print(f'Organizing dataset. Working on class {cls} ({idx+1:2d}/{len(classes_folders):2d}) and unique IDs ({idy+1:5d}/{len(unique_ids):5d})\r', end='')
            organized_files[cls][ids] = [f for f in class_files if f'-{ids}-' in f]

    print('\n')

    return organized_files

def get_AIIMS_organized_files(dataset_folder):
    '''
    Utility that given the path to the AIIMS dataset containing the folders of
    the different classes, organises the files in a dictionary nested as:
    class_name
        subject_ID or volume ID
            file name

    INPUT
    dataset_folder : str
            Path to the folder containing the folders of two classes (healthy
            and cancer)

    OUTPUT
    organized_files : dict
            A dictionary where the data is organized in hierarchical fashion:
            class_name
                subject_ID or volume ID
                    file name
    '''
    count_total_files = 0
    # get class names and create dictionary
    classes_folders = glob.glob(os.path.join(dataset_folder, '*/'))
    class_names = [os.path.basename(os.path.dirname(f)) for f in classes_folders]
    # fix names
    class_names = [c[0:c.find('_Sample')] for c in class_names]
    organized_files = dict.fromkeys(class_names)

    # loop through the classes, get all the unique subject ID or volumes IDs
    for (idx, cls), class_folder in zip(enumerate(organized_files.keys()), classes_folders):
        unique_ids_folder = glob.glob(os.path.join(class_folder,'*/'))
        unique_ids = [os.path.basename(os.path.dirname(f)) for f in unique_ids_folder]
        organized_files[cls] = dict.fromkeys(unique_ids)
        # get all the files for all the unique ids
        for idy, ids in enumerate(unique_ids):
            print(f'Organizing dataset. Working on class {cls} ({idx+1:2d}/{len(classes_folders):2d}) and unique IDs ({idy+1:5d}/{len(unique_ids):5d})\r', end='')
            organized_files[cls][ids] = glob.glob(os.path.join(class_folder, ids,'*'))
            count_total_files += len(organized_files[cls][ids])

    print(f'\n{count_total_files} total files')
    return organized_files

def get_per_image_train_test_val_split(organized_files,
                                    n_folds=1,
                                    n_per_class_test_imgs=250,
                                    n_per_class_val_imgs=250):
    '''
    Utility that given the organized files, splits the dataset into training,
    testing and validation based on the given configuration using a per_image
    split strategy.

    INPUT
    organized_files : dict
                dictionary containing the organized files in a hierarchical fashion
                obtained from the get_retinal_organized_files and get_AIIMS_organized_files
                functions.
    n_folds : int
            Number of folds to create on the training-validation pool
    n_per_class_test_imgs : int
            Number of images for each class in the test set
    n_per_class_val_imgs : int
            Number of images for each class in the validation set. This is enforced
            if only n_folds == 1. Otherwise, the number of images for every class
            depends on  n_folds

    OUTPUT
    per_fold_train_files : list
            List of lists containing the training filenames for each fold
    per_fold_val_files : list
            List of lists containing the validation filenames for each fold
    test_filenames : list
            List of filenames for the test files.
    '''
    # get test file names
    test_filenames = []
    test_used_class_files_index = {}
    for cls in organized_files.keys():
        all_class_files = []
        [all_class_files.extend(organized_files[cls][ids]) for ids in organized_files[cls].keys()]

        # random files
        idx_rnd = random.sample(range(len(all_class_files)), n_per_class_test_imgs)
        test_filenames.extend([all_class_files[i] for i in idx_rnd])

        # save index test files
        test_used_class_files_index[cls] = (idx_rnd)

    # work on the training and validation
    if n_folds > 1:
        # for every class, do cross validation
        per_fold_train_files = [[] for i in range(n_folds)]
        per_fold_val_files = [[] for i in range(n_folds)]
        kf = KFold(n_splits=n_folds)

        for cls in organized_files.keys():
            all_class_files = []
            [all_class_files.extend(organized_files[cls][ids]) for ids in organized_files[cls].keys()]
            for idx, (tr_idx, val_idx) in enumerate(kf.split(all_class_files)):
                per_fold_train_files[idx].extend([all_class_files[tr] for tr in tr_idx if tr not in test_used_class_files_index[cls]])
                per_fold_val_files[idx].extend([all_class_files[val] for val in val_idx if val not in test_used_class_files_index[cls]])
    else:
        # only one fold
        per_fold_train_files = [[] for i in range(n_folds)]
        per_fold_val_files = [[] for i in range(n_folds)]

        for cls in organized_files.keys():
            all_class_files = []
            [all_class_files.extend(organized_files[cls][ids]) for ids in organized_files[cls].keys()]
            # select validation files
            idx_rnd = random.sample(range(len(all_class_files)), n_per_class_val_imgs)
            per_fold_val_files[0].extend([all_class_files[i] for i in idx_rnd if i not in test_used_class_files_index[cls]])
            # put the remaining files as test
            per_fold_train_files[0].extend([all_class_files[i] for i in range(len(all_class_files)) if i not in test_used_class_files_index[cls]+idx_rnd])

    # shuffle file
    random.shuffle(test_filenames)
    for f in range(n_folds):
        random.shuffle(per_fold_train_files[f])
        random.shuffle(per_fold_val_files[f])

    return per_fold_train_files, per_fold_val_files, test_filenames


def get_per_volume_train_test_val_split(organized_files,
                                    n_folds=1,
                                    n_per_class_test_imgs=250,
                                    test_min_vol_per_class=2,
                                    n_per_class_val_imgs=250,
                                    val_min_vol_per_class=2):
    import functools
    import operator

    '''
    Utility that given the organized files in a nested dict as class/volume/files
    returns the test file selected per volume. The number of test files per
    class and the minimum number of volumes from which to take the images can be
    specified.

    INPUT
    organized_files : dict
                dictionary containing the organized files in a hierarchical fashion
                obtained from the get_retinal_organized_files and get_AIIMS_organized_files
                functions.
    n_folds : int
            Number of folds to create on the training-validation pool
    n_per_class_test_imgs : int
            Number of images for each class in the test set
    test_min_vol_per_class : int
            Minimum number of different volumes from where the test files are
            taken from.
    n_per_class_val_imgs : int
            Number of images for each class in the validation set. This is enforced
            if only n_folds == 1. Otherwise, the number of images for every class
            depends on  n_folds
    val_min_vol_per_class : int
            Minimum number of different volumes from where the validation files
            are taken from.

    OUTPUT
    per_fold_train_files : list
            List of lists containing the training filenames for each fold
    per_fold_val_files : list
            List of lists containing the validation filenames for each fold
    test_filenames : list
            List of filenames for the test files.
    summary_ids : dict
            Dictionary summarising which volume are included in the training,
            validation and testing sets.
    '''

    test_filenames = []
    used_test_ids = {}

    for cls in organized_files.keys():
        used_test_ids[cls] = []
        aus_cls_files = []
        aus_used_t_ids = []
        while not all([len(aus_used_t_ids)>=test_min_vol_per_class, len(aus_cls_files) >= n_per_class_test_imgs]):
            # randomly select an id
            t_id = random.choice(list(organized_files[cls].keys()))
            # check the the selected id is not already used. if not add images as test
            if t_id not in used_test_ids[cls]:
                aus_used_t_ids.append(t_id)
                aus_cls_files.extend(organized_files[cls][t_id])

        # since there can be more than n_per_class_test_imgs*len(organized_files.keys()) images, select precisely
        # test_filenames.extend(random.sample(aus_cls_files, n_per_class_test_imgs))
        test_filenames.extend(aus_cls_files)
        used_test_ids[cls].extend(aus_used_t_ids)

    # work on the training and validation
    used_val_ids = {}
    used_tr_ids = {}

    if n_folds > 1:
        # for every class, do cross validation
        per_fold_train_files = [[] for i in range(n_folds)]
        per_fold_val_files = [[] for i in range(n_folds)]
        kf = KFold(n_splits=n_folds)

        for cls in organized_files.keys():
            # take out all the IDs for this class that were used for testing
            remaining_IDs = [ids for ids in organized_files[cls].keys() if ids not in used_test_ids[cls]]
            used_val_ids[cls] = []
            used_tr_ids[cls] = []

            for idx, (tr_ids, val_ids) in enumerate(kf.split(remaining_IDs)):
                aus_tr_files = [organized_files[cls][remaining_IDs[i]] for i in tr_ids]
                per_fold_train_files[idx].extend(functools.reduce(operator.concat, aus_tr_files))
                used_tr_ids[cls].append([remaining_IDs[i] for i in tr_ids])

                aus_val_files = [organized_files[cls][remaining_IDs[i]] for i in val_ids]
                per_fold_val_files[idx].extend(functools.reduce(operator.concat, aus_val_files))
                used_val_ids[cls].append([remaining_IDs[i] for i in val_ids])
    else:
        # only one fold. Take n_per_class_val_imgs from val_min_vol_per_class
        per_fold_train_files = [[] for i in range(n_folds)]
        per_fold_val_files = [[] for i in range(n_folds)]

        for cls in organized_files.keys():
            aus_cls_files = []
            aus_used_t_ids = []

            used_val_ids[cls] = []
            used_tr_ids[cls] = []

            while not all([len(aus_used_t_ids)>=val_min_vol_per_class, len(aus_cls_files) >= n_per_class_val_imgs]):
                # randomly select an id
                t_id = random.choice(list(organized_files[cls].keys()))
                # check the the selected id is not already used. if not add images as test
                if t_id not in used_test_ids[cls]:
                    aus_used_t_ids.append(t_id)
                    aus_cls_files.extend(organized_files[cls][t_id])
                    used_val_ids.append()

            used_val_ids[cls].append(aus_used_t_ids)
            used_tr_ids[cls].append([i for i in organized_files[cls].keys() if i not in used_test_ids[cls]] and i not in aus_used_t_ids)

            # since there can be more images, select precisely n_per_class_val_imgs
            per_fold_val_files[0].extend(random.sample(aus_cls_files, n_per_class_val_imgs))
            # update list of used IDs #####################################
            used_test_ids[cls].extend(aus_cls_files)
            # take care of the training data
            aus_train_files = [organized_files[cls][i] for i in organized_files[cls].keys() if i not in used_test_ids[cls]]
            per_fold_train_files[0].extend(functools.reduce(operator.concat, aus_train_files))


    # shuffle file
    random.shuffle(test_filenames)
    for f in range(n_folds):
        random.shuffle(per_fold_train_files[f])
        random.shuffle(per_fold_val_files[f])

    summary_ids = {
                    'test':used_test_ids,
                    'train':used_tr_ids,
                    'val':used_val_ids}

    return per_fold_train_files, per_fold_val_files, test_filenames, summary_ids


## parse inline parameters

# parser = argparse.ArgumentParser(description='Script that runs a cross-validation training for OCT 2D image classification.')
# parser.add_argument('-wd','--working_directory' ,required=False, help='Provide the Working Directory where the models_tf.py, utilities.py and utilities_models_tf.py files are. This folder will also be the one where the trained models will be saved. If not provided, the current working directory is used', default=os.getcwd())
# # dataset parameters
# parser.add_argument('-df', '--dataset_folder', required=True, help='Provide the Dataset Folder where the downloaded dataset are located. Note that for the OCT2017 dataset in case of per_image or per_volume split strategy, the images should be first reorganized using the refine_dataset.py script. Give the original dataset path in case the original_split strategy is set.')
# parser.add_argument('-dt', '--dataset_type', required=True, help='Specifies which dataset (retinal or AIIMS) is given for training. This will be used to set the appropriate dataloader function')
# parser.add_argument('-dss', '--dataset_split_strategy', required=False, help='Specifies the strategy used to split the detaset into training testing and validation. Three options available per_volume, per_image (innapropriate splitting) or original (only OCT2017)', default='per_volume')
# parser.add_argument('-ids', '--imbalance_data_strategy', required=False, help='Strategy to use to tackle imbalance data. Available none or weights', default='weights')
# # model parameters adn training parameters
# parser.add_argument('-mc', '--model_configuration', required=False, help='Provide the Model Configuration (LightOCT or others if implemented in the models_tf.py file).', default='LightOCT')
# parser.add_argument('-mn', '--model_name', required=False, help='Provide the Model Name. This will be used to create the folder where to save the model. If not provided, the current datetime will be used', default=datetime.now().strftime("%H:%M:%S"))
# parser.add_argument('-f', '--folds', required=False, help='Number of folds. Default is 3', default='3')
# parser.add_argument('-l', '--loss', required=False, help='Loss to use to train the model (cce or wcce). Default is cce', default='cce')
# parser.add_argument('-lr', '--learning_rate', required=False, help='Learning rate.', default=0.001)
# parser.add_argument('-bs', '--batch_size', required=False, help='Batch size.', default=50)
# parser.add_argument('-ks', '--kernel_size', nargs='+', required=False, help='Encoder conv kernel size.', default=(5,5))
# parser.add_argument('-augment', '--augmentation', required=False, help='Specify if data augmentation is to be performed (True) or not (False)', default=True)
# # debug parametres
# parser.add_argument('-v', '--verbose',required=False, help='How much to information to print while training: 0 = none, 1 = at the end of an epoch, 2 = detailed progression withing the epoch.', default=0.1)
# parser.add_argument('-ctd', '--check_training', required=False, help='If True, checks that none of the test images is in the training/validation set. This may take a while depending on the size of the dataset.', default=True)
# parser.add_argument('-db', '--debug', required=False, help='True if want to use a smaller portion of the dataset for debugging', default=False)
#
# args = parser.parse_args()
#
# # # # # # # # # # # # # # # # parse variables
# working_folder = args.working_directory
# # dataset variables
# dataset_folder = args.dataset_folder
# dataset_type = args.dataset_type
# dataset_split_strategy = args.dataset_split_strategy
# imbalance_data_strategy = args.imbalance_data_strategy
# # model parameters
# model_configuration = args.model_configuration
# model_save_name = args.model_name
# loss = args.loss
# learning_rate = float(args.learning_rate)
# batch_size = int(args.batch_size)
# data_augmentation = args.augmentation
# N_FOLDS = int(args.folds)
# kernel_size = [int(i) for i in args.kernel_size]
# # debug variables
# verbose = int(args.verbose)
# debug = args.debug == 'True'
# check_training = args.check_training == 'True'


# # # # # # # # # #  parse variables
working_folder = "/flush/iulta54/Research/P3_1-OCT_DATASET_STUDY"
# dataset variables
# dataset_folder = "/flush/iulta54/Research/Data/OCT/Retinal/Zhang_dataset_version_3/OCT"
dataset_folder = "/flush/iulta54/Research/Data/OCT/Retinal/Zhang_dataset/per_class_files"
# dataset_folder = "/flush/iulta54/Research/Data/OCT/AIIMS_Dataset/original"
dataset_type = 'retinal'
dataset_split_strategy = 'per_image'
imbalance_data_strategy = 'none'
# model parameters
model_configuration = 'LightOCT'
model_save_name = 'TEST'
loss = 'cce'
learning_rate = 0.001
batch_size = 256
data_augmentation = True
N_FOLDS = 5
kernel_size = (3,3)
# debug variables
verbose = 2
debug = False
check_training = False


# check if working folder and dataset folder exist
if os.path.isdir(working_folder):
    # check if the trained_model folder exists, if not create it
    if not os.path.isdir(os.path.join(working_folder, 'trained_models')):
        print('trained_model folders does not exist in the working path, creating it...')
        save_path = os.path.join(working_folder, 'trained_models')
        os.mkdir(save_path)
else:
    print('The provided working folder does not exist. Input a valid one. Given {}'.format(working_folder))
    sys.exit()

if not os.path.isdir(dataset_folder):
    print(f'The dataset folder provided does not exist. Input a valid one. Given {dataset_folder}')
    sys.exit()

if debug:
    print(f'\n{"-"*70}')
    print(f'{"Configuration file script - running in debug mode"}')
    print(f'{"-"*70}\n')
else:
    print(f'\n{"-"*25}')
    print(f'{"Configuration file script"}')
    print(f'{"-"*25}\n')

print(f'{"Working directory":<26s}: {working_folder}\n')
print(f'{"Dataset folder":<26s}: {dataset_folder}')
print(f'{"Dataset type":<26s}: {dataset_type}')
print(f'{"Dataset split strategy":<26s}: {dataset_split_strategy} \n')

print(f'{"Model configuration":<26s}: {model_configuration}')
print(f'{"Model save name":<26s}: {model_save_name}')
print(f'{"Loss function":<26s}: {loss}')
print(f'{"Learning rate":<26s}: {learning_rate}')
print(f'{"Batch size":<26s}: {batch_size}')
print(f'{"Data augmentation":<26s}: {data_augmentation} ')

# import local utilities
import utilities

## get file names organised as class/subject-volume/file
importlib.reload(utilities)

if dataset_type == 'retinal':
    if any([dataset_split_strategy=='per_volume', dataset_split_strategy=='per_image']):
        # heuristic for the retinal dataset only if not using the original split
        organized_files = get_retinal_organized_files(dataset_folder)

        unique_labels = list(organized_files.keys())
        nClasses =len(unique_labels)
elif dataset_type == 'AIIMS':
    # heuristic for the AIIMS dataset
    organized_files = utilities.get_AIIMS_organized_files(dataset_folder)

    unique_labels = list(organized_files.keys())
    nClasses =len(unique_labels)
else:
    raise ValueError(f'The dataset type is not one of the available one. Expected retinal or AIIMS but give {dataset_type}')

## set training, validation and testing sets
n_per_class_test_imgs = 50
n_per_class_val_imgs = 100
test_min_vol_per_class = 2
val_min_vol_per_class = 2

# fix random seed
random.seed(29122009)

if dataset_split_strategy == 'per_volume':
    per_fold_train_files, per_fold_val_files, test_file, summary_ids = get_per_volume_train_test_val_split(organized_files,
                                        n_folds=N_FOLDS,
                                        n_per_class_test_imgs=n_per_class_test_imgs,
                                        test_min_vol_per_class=test_min_vol_per_class,
                                        n_per_class_val_imgs=n_per_class_val_imgs,
                                        val_min_vol_per_class=val_min_vol_per_class)
    # print summary ids if in debug mode
    if debug:
        for f in range(N_FOLDS):
            print(f'Fold {f:2d}')
            for cls in organized_files.keys():
                aus_str = f'    Class {cls:{max([len(c) for c in organized_files.keys()])}s}'
                print(f'{aus_str} - tr : {summary_ids["train"][cls][f]}')
                print(f'{" ":{len(aus_str)}s} - val: {summary_ids["val"][cls][f]}')
        # print testing
        print('Test IDs')
        for cls in organized_files.keys():
            print(f'Class {cls:{max([len(c) for c in organized_files.keys()])}s}: {summary_ids["test"][cls]}')

elif dataset_split_strategy == 'per_image':
    per_fold_train_files, per_fold_val_files, test_file = get_per_image_train_test_val_split(organized_files,
                                        n_folds=N_FOLDS,
                                        n_per_class_test_imgs=n_per_class_test_imgs,
                                        n_per_class_val_imgs=n_per_class_val_imgs)

elif all([dataset_type == 'retinal', dataset_split_strategy == 'original']):
    # get the training and test images from the original downloaded folder.
    # Using the test files as validation
    per_fold_train_files = [glob.glob(os.path.join(dataset_folder, 'train','*','*'))]
    per_fold_val_files = [glob.glob(os.path.join(dataset_folder, 'test','*','*'))]
    test_file = per_fold_val_files[0]

    # given the original split, setting the fold number to 1
    N_FOLDS = 1
    # get the number of classes
    unique_labels = [os.path.basename(f) for f in glob.glob(os.path.join(dataset_folder, 'train','*'))]
    nClasses =len(unique_labels)
else:
    raise ValueError(f'The given dataset split strategy is not valid. Expected per_image, per_volume or original (original only for retinal dataset). Give {dataset_split_strategy} for dataset type {dataset_type}')

# setting class weights to 1
class_weights = [1] * nClasses

for f in range(N_FOLDS):
    print(f'Fold {f+1}: training on {len(per_fold_train_files[f]):5d} and validation on {len(per_fold_val_files[f]):5d}')
print(f'Testing files: {len(test_file)}')
print(f'{"Class weights":<10s}: {class_weights}')



# ############## check that no testing files are in the training validation pool
if check_training:
    print('Checking if any test samples are in the training - validation pool (this may take time...)')
    duplicated = []
    for idx, ts in enumerate(test_file):
        print(f'Checked {idx+1}/{len(test_file)} \r', end='')
        for train_file, val_file in zip(per_fold_train_files, per_fold_val_files):
            for tr in train_file+val_file:
                # get subject/volume ID for check
                if dataset_type == 'AIIMS':
                    ts_id = pathlib.Path(ts).parts[-2]
                    tr_id = pathlib.Path(tr).parts[-2]
                elif dataset_type == 'retinal':
                    ts_id = os.path.join(pathlib.Path(ts).parts[-2],
                                         pathlib.Path(ts).parts[-1])
                    ts_id = ts_id[0:ts_id.rfind('-')]

                    tr_id = os.path.join(pathlib.Path(tr).parts[-2],
                                         pathlib.Path(tr).parts[-1])
                    tr_id = tr_id[0:tr_id.rfind('-')]
                print(f'Checked {idx+1}/{len(test_file)} ({ts_id}/{tr_id})\r', end='')
                if tr_id == ts_id:
                    duplicated.append(ts)
                    raise ValueError(f'Some of the testing files are in the trianing - validation pool ({len(duplicated)} out of {len(test_file)}). CHECK IMPLEMENTATION!!!')
    print('No testing files found in the training - validation pool. All good!!!')
else:
    print(f'\n {"¤"*10} \n ATTENTION! Not checking if test images are in the training/validation pool. \n Use with care!!! \n {"¤"*10}')

## Save all the information in a configuration file
'''
The configuration file will be used by the training routine to access the
the train-val-test files as well as the different set-up for the model. Having a
separate configuration file helps keeping the training routine more clean.
'''
print(f'\nSaving configuration file...')

json_dict = OrderedDict()
json_dict['working_folder'] = working_folder
json_dict['dataset_folder'] = dataset_folder
json_dict['dataset_type'] = dataset_type
json_dict['dataset_split_strategy'] = dataset_split_strategy

json_dict['unique_labels'] = unique_labels

json_dict['model_configuration'] = model_configuration
json_dict['model_save_name'] = model_save_name
json_dict['loss'] = loss
json_dict['learning_rate'] = learning_rate
json_dict['batch_size'] = batch_size
json_dict['input_size'] = [240, 400] if dataset_type=='retinal' else [400, 240]
json_dict['n_channels'] = 1 if dataset_type=='retinal' else 3
json_dict['kernel_size'] = kernel_size
json_dict['data_augmentation'] = data_augmentation

json_dict['N_FOLDS'] = N_FOLDS
json_dict['verbose'] = verbose
json_dict['imbalance_data_strategy'] = imbalance_data_strategy

json_dict['training'] = per_fold_train_files
json_dict['validation'] = per_fold_val_files
json_dict['test'] = test_file
json_dict['class_weights'] = list(class_weights)

# save file
save_model_path = os.path.join(working_folder, 'trained_models', model_save_name)

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

json_dict['save_model_path'] = save_model_path


with open(os.path.join(save_model_path,'config.json'), 'w') as fp:
    json.dump(json_dict, fp)

print(f'Configuration file created. Avvailable at {save_model_path}')






