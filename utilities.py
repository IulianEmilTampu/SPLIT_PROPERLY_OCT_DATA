import os
import cv2
import math
import glob
import time
import numbers
import random
import itertools
import numpy as np
import pathlib
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, RepeatedKFold

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

## UTILITIES FOR ORGANISING THE FILES

def get_Kermany_organized_files(dataset_folder):
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

def get_Srinivas_organized_files(dataset_folder):
    '''
    Utility that given the path to the Srinivas dataset containing the folders of
    the different subjects, organises the files in a dictionary nested as:
    class_name
        subject_ID or volume ID
            file name

    INPUT
    dataset_folder : str
            Path to the folder containing the folders of the different subjects

    OUTPUT
    organized_files : dict
            A dictionary where the data is organized in hierarchical fashion:
            class_name
                subject_ID or volume ID
                    file name
    '''
    count_total_files = 0
    # get class names and create dictionary
    class_names = []
    subjects_folders = glob.glob(os.path.join(dataset_folder, '*'))
    for subj_folder in subjects_folders:
        '''
        The folders in the dataset are named ClassNameNumber. We only want the ClassName
        thus, here we find the index of the first not digit in the folder basename and
        use that to take the ClassName
        '''
        indx = [s.isdigit() for s in os.path.basename(subj_folder)].index(True)
        class_names.append(os.path.basename(subj_folder)[0:indx])

    organized_files = dict.fromkeys(class_names)

    # loop through all the classes and get the file names of all the images belonging to all the subjects
    for idx, cls in enumerate(organized_files.keys()):
        # loop through all the subject folders and, if belonging to the class, get the file names
        organized_files[cls] = {}
        for subj_folder in subjects_folders:
            if cls in os.path.basename(subj_folder):
                print(f'Organizing dataset. Working on class {cls} \r', end='')
                subj_id = os.path.basename(subj_folder)
                organized_files[cls][subj_id] = glob.glob(os.path.join(subj_folder, 'TIFFs', '8bitTIFFs','*'))
                count_total_files += len(organized_files[cls][subj_id])
                
    print(f'\n{count_total_files} total files')
    return organized_files

def get_per_image_train_test_val_split(organized_files,
                                    n_folds=1,
                                    n_repetitions_cv=1, 
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
        per_fold_train_files = [[] for i in range(n_folds*n_repetitions_cv)]
        per_fold_val_files = [[] for i in range(n_folds*n_repetitions_cv)]
        rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repetitions_cv, random_state=2652124)

        for cls in organized_files.keys():
            all_class_files = []
            [all_class_files.extend(organized_files[cls][ids]) for ids in organized_files[cls].keys()]
            for idx, (tr_idx, val_idx) in enumerate(rkf.split(all_class_files)):
                per_fold_train_files[idx].extend([all_class_files[tr] for tr in tr_idx if tr not in test_used_class_files_index[cls]])
                per_fold_val_files[idx].extend([all_class_files[val] for val in val_idx if val not in test_used_class_files_index[cls]])

    else:
        # only one fold
        per_fold_train_files = [[] for i in range(n_folds*n_repetitions_cv)]
        per_fold_val_files = [[] for i in range(n_folds*n_repetitions_cv)]

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
    for f in range(n_folds*n_repetitions_cv):
        random.shuffle(per_fold_train_files[f])
        random.shuffle(per_fold_val_files[f])

    return per_fold_train_files, per_fold_val_files, test_filenames


def get_per_volume_train_test_val_split(organized_files,
                                    n_folds=1,
                                    n_repetitions_cv=1,
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

    # work on the test set
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
        test_filenames.extend(random.sample(aus_cls_files, n_per_class_test_imgs))
        # test_filenames.extend(aus_cls_files)
        used_test_ids[cls].extend(aus_used_t_ids)

    # work on the training and validation sets
    used_val_ids = {}
    used_tr_ids = {}

    if n_folds > 1:
        # for every class, do cross validation
        per_fold_train_files = [[] for i in range(n_folds*n_repetitions_cv)]
        per_fold_val_files = [[] for i in range(n_folds*n_repetitions_cv)]
        rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repetitions_cv, random_state=2652124)

        for cls in organized_files.keys():
            # take out all the IDs for this class that were used for testing
            remaining_IDs = [ids for ids in organized_files[cls].keys() if ids not in used_test_ids[cls]]
            used_val_ids[cls] = []
            used_tr_ids[cls] = []

            for idx, (tr_ids, val_ids) in enumerate(rkf.split(remaining_IDs)):
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


    # shuffle files
    random.shuffle(test_filenames)
    for f in range(n_folds*n_repetitions_cv):
        random.shuffle(per_fold_train_files[f])
        random.shuffle(per_fold_val_files[f])

    summary_ids = {
                    'test':used_test_ids,
                    'train':used_tr_ids,
                    'val':used_val_ids}

    return per_fold_train_files, per_fold_val_files, test_filenames, summary_ids


## TENSORFLOW DATA GENERATOR for RETINAL DATASET

def Kermany_data_gen(img_files,
                unique_labels=None,
                batch_size=1,
                training=True,
                channels=1,
                input_size=(240, 400),
                normalize=True,
                normalization_strategy=1,
                categorical=False, 
                random_label_experiment=False, 
                random_label_experiment_seed=291209,):
    '''
    Script that uses the file names of Kermany OCT 2D images for tissue classification
    to generate a tf dataset ready for training, validation or test

    INPUT
    image_files : list
        List of strings of path to the image files
    unique_labels : list
        List of strings specifying the unique labels in the dataset. If not given
        the unique labels will be infered from the folder where the files are
        located.
    batch_size : int
    training : bool
        True if the dataset will be used for training. This will shuffle the dataset
        as well as prefetch it.
    channels : int
        Number of channels of the input images
    input_size : tuple
        Desired size of the images to be outputed by the generator
    normalize : bool
        If to perform normalization or not.
    normalization_strategy : int
        Value used to select the type of normalization. 0 for [0,1] or 1 for
        [-1,1] normalization
    categorical : bool
        If true the labels will be outputed as categorical.

    OUTPUT

    dataset : tf dataset
        A tf dataset with the first instance the image and the second the labels.
    '''

    def str_to_integer_label(str_labels, unique_labels=None):
        '''
        Converts a list of string based labels to integers
        '''
        if unique_labels is None:
            unique_labels = list(set(str_labels))
            unique_labels.sort()

        int_labels = [unique_labels.index(l) for l in str_labels]
        return int_labels

    # get string labels from the file names (assuming .../LABEL/LABEL_0001.extension)
    str_labels = [os.path.basename(os.path.dirname(f)) for f in img_files]
    # convert to integer labels
    int_labels = str_to_integer_label(str_labels)
    # shuffle labels if the random_label_experiment is to be performed
    if random_label_experiment:
        random.Random(random_label_experiment_seed).shuffle(int_labels)
    # convert to categorical if needed
    if categorical:
        int_labels = tf.one_hot(int_labels, len(unique_labels))

    # create tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(img_files), tf.constant(int_labels)))

    # actually open the images
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.io.decode_jpeg(tf.io.read_file(x), channels=channels), y)))

    # crop to specified input size
    # dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.crop_to_bounding_box(x, offset_height=61, offset_width=110, target_height=input_size[0], target_width=input_size[1]),y)))
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.resize(x,size=input_size, method=tf.image.ResizeMethod.BILINEAR),y)))

    # set up dataset
    AUTOTUNE = tf.data.AUTOTUNE
    if training is True:
        dataset = dataset.shuffle(1000)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)

    if normalize:
        if normalization_strategy == 1:
            # normalize images [-1, 1]
            normalizer = layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
            dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (normalizer(x), y)))
        elif normalization_strategy == 0:
            # normalize images [0, 1]
            normalizer = layers.experimental.preprocessing.Rescaling(1./255, offset=0)
            dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (normalizer(x), y)))
        else:
            print(f'Unknown normalization strategy. Given {normalization_strategy} expected 0 for [0,1] or 1 for [-1,1] normalization')
            sys.exit()

    return dataset

## TENSORFLOW DATA GENERATOR for AIIMS dataset

def AIIMS_data_gen(img_files,
            unique_labels=None,
            batch_size=1,
            training=True,
            channels=3,
            input_size=(400, 240),
            normalize=True,
            normalization_strategy=1,
            categorical=False,
            random_label_experiment=False, 
            random_label_experiment_seed=291209,):
    '''
    Script that uses the file names of the AIIMS OCT 2D images for tissue classification
    to generate a tf dataset ready for training, validation or test

    INPUT
    image_files : list
        List of strings of path to the image files
    unique_labels : list
        List of strings specifying the unique labels in the dataset. If not given
        the unique labels will be infered from the folder where the files are
        located.
    batch_size : int
    training : bool
        True if the dataset will be used for training. This will shuffle the dataset
        as well as prefetch it.
    channels : int
        Number of channels of the input images
    input_size : tuple
        Desired size of the images to be outputed by the generator
    normalize : bool
        If to perform normalization or not.
    normalization_strategy : int
        Value used to select the type of normalization. 0 for [0,1] or 1 for
        [-1,1] normalization
    categorical : bool
        If true the labels will be outputed as categorical.

    OUTPUT

    dataset : tf dataset
        A tf dataset with the first instance the image and the second the labels.
    '''

    # get string labels from the file names (assuming .../LABEL/Subject/LABEL_0001.extension)
    str_labels = [pathlib.Path(f).parts[-3] for f in img_files]
    def str_to_integer_label(str_labels, unique_labels=None):
        '''
        Converts a list of string based labels to integers
        '''
        if unique_labels is None:
            unique_labels = list(set(str_labels))
            unique_labels.sort()

        int_labels = [unique_labels.index(l) for l in str_labels]
        return int_labels

    # convert to integer labels
    int_labels = str_to_integer_label(str_labels)

    # shuffle labels if the random_label_experiment is to be performed
    if random_label_experiment:
       random.Random(random_label_experiment_seed).shuffle(int_labels)

    if categorical:
        int_labels = tf.one_hot(int_labels, len(unique_labels))

    # create tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(img_files), tf.constant(int_labels)))

    # actually open the images
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.io.decode_bmp(tf.io.read_file(x), channels=channels), y)))
    # convert to gray scale
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.rgb_to_grayscale(x), y)))


    # crop to specified input size
    # dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.crop_to_bounding_box(x, offset_height=61, offset_width=110, target_height=input_size[0], target_width=input_size[1]),y)))
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.resize(x,size=input_size, method=tf.image.ResizeMethod.BILINEAR),y)))

    if normalize:
        '''
        There is a problem with the normalization sice the scale bar is perfect white while the overll images
        do not have such value. This squeezes the histogram of the normalized image to be
        very small. Need to normalize using quantiles.
        Quantile values are computed on the test detaset and correspond to 2% and
        98% quantiles.
        '''
        # quantile values obtained on the training set
        q_min = 0
        q_max = 88
        if normalization_strategy == 1:
            # normalize images [-1, 1]
            dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (2 * (x-q_min) / (q_max-q_min) - 1, y)))

        elif normalization_strategy == 0:
            # normalize images [0, 1]
            normalizer = layers.experimental.preprocessing.Rescaling(1./255, offset=0)
            dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (normalizer(x), y)))
        else:
            print(f'Unknown normalization strategy. Given {normalization_strategy} expected 0 for [0,1] or 1 for [-1,1] normalization')
            sys.exit()

    # set up dataset
    AUTOTUNE = tf.data.AUTOTUNE
    if training is True:
        dataset = dataset.shuffle(5000)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset

def Srinivas_data_gen(img_files,
            unique_labels=None,
            batch_size=1,
            training=True,
            channels=3,
            input_size=(768,496),
            normalize=True,
            normalization_strategy=1,
            categorical=False,
            random_label_experiment=False, 
            random_label_experiment_seed=291209,):
    '''
    Script that uses the file names of Srinivas OCT 2D images for tissue classification
    to generate a tf dataset ready for training, validation or test

    INPUT
    image_files : list
        List of strings of path to the image files
    unique_labels : list
        List of strings specifying the unique labels in the dataset. If not given
        the unique labels will be infered from the folder where the files are
        located.
    batch_size : int
    training : bool
        True if the dataset will be used for training. This will shuffle the dataset
        as well as prefetch it.
    channels : int
        Number of channels of the input images
    input_size : tuple
        Desired size of the images to be outputed by the generator
    normalize : bool
        If to perform normalization or not.
    normalization_strategy : int
        Value used to select the type of normalization. 0 for [0,1] or 1 for
        [-1,1] normalization
    categorical : bool
        If true the labels will be outputed as categorical.

    OUTPUT

    dataset : tf dataset
        A tf dataset with the first instance the image and the second the labels.
    '''

    # get string labels from the file names (assuming .../LabelNmber/TIFFs/8bitTIFFs/file.extension)
    str_labels = [pathlib.Path(f).parts[-4] for f in img_files]
    class_names = []
    for label_name in str_labels:
        indx = [s.isdigit() for s in label_name].index(True)
        class_names.append(label_name[0:indx])

    def str_to_integer_label(str_labels, unique_labels=None):
        '''
        Converts a list of string based labels to integers
        '''
        if unique_labels is None:
            unique_labels = list(set(str_labels))
            unique_labels.sort()

        int_labels = [(unique_labels.index(l)) for l in str_labels]
        return int_labels

    # convert to integer labels
    int_labels = str_to_integer_label(class_names)

    # shuffle labels if the random_label_experiment is to be performed
    if random_label_experiment:
       random.Random(random_label_experiment_seed).shuffle(int_labels)

    if categorical:
        int_labels = tf.one_hot(int_labels, len(unique_labels))

    # create tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(img_files), tf.constant(int_labels)))

    # actually open the images
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.cast(tfio.experimental.image.decode_tiff(tf.io.read_file(x), index=0),tf.float32), y)))
    # convert to gray scale
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.rgb_to_grayscale(x[:,:,0:-1]), y)))


    # crop to specified input size
    # dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.crop_to_bounding_box(x, offset_height=61, offset_width=110, target_height=input_size[0], target_width=input_size[1]),y)))
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (tf.image.resize(x,size=input_size, method=tf.image.ResizeMethod.BILINEAR),y)))


    if normalize:
        '''
        There is a problem with the normalization sice the scale bar is perfect white while the overll images
        do not have such value. This squeezes the histogram of the normalized image to be
        very small. Need to normalize using quantiles.
        Quantile values are computed on the test detaset and correspond to 2% and
        98% quantiles.
        '''
        # quantile values obtained on the training set
        q_min = 0
        q_max = 250
        if normalization_strategy == 1:
            # normalize images [-1, 1]
            dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (2 * (x-q_min) / (q_max-q_min) - 1, y)))

        elif normalization_strategy == 0:
            # normalize images [0, 1]
            normalizer = layers.experimental.preprocessing.Rescaling(1./255, offset=0)
            dataset = dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (normalizer(x), y)))
        else:
            print(f'Unknown normalization strategy. Given {normalization_strategy} expected 0 for [0,1] or 1 for [-1,1] normalization')
            sys.exit()

    # set up dataset
    AUTOTUNE = tf.data.AUTOTUNE
    if training is True:
        dataset = dataset.shuffle(5000)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


## CONFISION MATRIX

def plotConfusionMatrix(GT, PRED, classes, Labels=None, cmap=plt.cm.Blues, savePath=None, saveName=None, draw=False):
    '''
    Funtion that plots the confision matrix given the ground truths and the predictions
    '''
    # compute confusion matrix
    cm = confusion_matrix(GT, PRED)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation=None, cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = cm.max()/2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if cm[i,j] > thresh else 'black',
            fontsize=25)

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Prediction', fontsize=15)

    acc = 100*(np.trace(cm) / np.sum(cm))
    plt.title('Confusion matrix -> ' + 'Accuracy {:05.2f}'.format(acc), fontsize=20)
    fig.tight_layout()

    # save if needed
    if savePath is not None:
        # set up name
        if saveName is None:
            saveName = "ConfisionMatrix_ensemble_prediction"

        if os.path.isdir(savePath):
            fig.savefig(os.path.join(savePath, f'{saveName}.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(savePath, f'{saveName}.png'), bbox_inches='tight', dpi = 100)
        else:
            raise ValueError('Invalid save path: {}'.format(os.path.join(savePath, f'{saveName}')))

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return acc

## PLOT ROC

def plotROC(GT, PRED, classes, savePath=None, saveName=None, draw=False):
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    from scipy import interp
    from tensorflow.keras.utils import to_categorical
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    '''
    Funtion that plots the ROC curve given the ground truth and the logits prediction

    INPUT
    GT : array
        True labels
    PRED : array
        aRRAY of float the identifies the logits prediction
    classes : list
        lIST of string that identifies the labels of each class
    save path : string
        Identifies the path where to save the ROC plots
    save name : string
        Specifying the name of the file to be saved
    draw : bool
        True if to print the ROC curve

    OUTPUT
    fpr : dictionary that contains the false positive rate for every class and
           the overall micro and marco averages
    trp : dictionary that contains the true positive rate for every class and
           the overall micro and marco averages
    roc_auc : dictionary that contains the area under the curve for every class and
           the overall micro and marco averages

    Check this link for better understanding of micro and macro-averages
    https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

    Here computing both the macro-average ROC and the micro-average ROC.
    Using code from https://scikit-learn.org/dev/auto_examples/model_selection/plot_roc.html with modification
    '''
    # define variables
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)
    lw = 2 # line width

    # make labels categorical
    GT = to_categorical(GT, num_classes=n_classes)

    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(GT[:,i], PRED[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(GT.ravel(), PRED.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # ¤¤¤¤¤¤¤¤¤¤ macro-average roc

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves and save
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(classes[i], roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=lw)

    major_ticks = np.arange(0, 1, 0.1)
    minor_ticks = np.arange(0, 1, 0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    plt.grid(color='b', linestyle='-.', linewidth=0.1, which='both')

    ax.set_xlabel('False Positive Rate', fontsize=25)
    ax.set_ylabel('True Positive Rate', fontsize=25)
    ax.set_title('Multi-class ROC (OneVsAll)', fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ work on the zummed-in image
    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    axins = zoomed_inset_axes(ax, zoom=1, loc=7, bbox_to_anchor=(0,0,0.99,0.9), bbox_transform=ax.transAxes)

    axins.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    axins.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(n_classes), colors):
        axins.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {} (area = {:0.2f})'
                ''.format(classes[i], roc_auc[i]))

        # sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.3, 0.7, 1.0
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.grid(color='b', linestyle='--', linewidth=0.1)

        axins.set_xticks(np.linspace(x1, x2, 4))
        axins.set_yticks(np.linspace(y1, y2, 4))

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=3, fc='none', ec='0.5', ls='--')

    # save is needed
    if savePath is not None:
        # set up name
        if saveName is None:
            saveName = "Multiclass_ROC"

        if os.path.isdir(savePath):
            fig.savefig(os.path.join(savePath, f'{saveName}.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(savePath, f'{saveName}.png'), bbox_inches='tight', dpi = 100)
        else:
            raise ValueError('Invalid save path: {}'.format(os.path.join(savePath, f'{saveName}')))

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return fpr, tpr, roc_auc

## PLOR PR (precision and recall) curves

def plotPR(GT, PRED, classes, savePath=None, saveName=None, draw=False):
    from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
    from sklearn.metrics import average_precision_score
    from itertools import cycle
    from scipy import interp
    from tensorflow.keras.utils import to_categorical
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    '''
    Funtion that plots the PR (precision and recall) curve given the ground truth and the logits prediction

    INPUT
    - GT: true labels
    - PRED: array of float the identifies the logits prediction
    - classes: list of string that identifies the labels of each class
    - save path: sting that identifies the path where to save the ROC plots
    - save name: string the specifies the name of the file to be saved.
    - draw: bool if to print or not the ROC curve

    OUTPUT
    - precision: dictionary that contains the precision every class and micro average
    - recall: dictionary that contains the recall for every class and micro average
    - average_precision: float of the average precision
    - F1: dictionare containing the micro and marco average f1-score
    '''
    # define variables
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(classes)
    lw = 2 # line width

    # ¤¤¤¤¤¤¤¤¤¤¤ f1_score
    F1 = {
        'micro':f1_score(GT, np.argmax(PRED, axis=-1), average='micro'),
        'macro':f1_score(GT, np.argmax(PRED, axis=-1), average='macro')
    }
    # print('F1-score (micro and macro): {0:0.2f} and {0:0.2f}'.format(F1['micro'], F1['macro']))

    # make labels categorical
    GT = to_categorical(GT, num_classes=n_classes)

    # ¤¤¤¤¤¤¤¤¤¤¤ micro-average roc
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(GT[:,i], PRED[:,i])
        average_precision[i] = average_precision_score(GT[:,i], PRED[:,i])


    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(GT.ravel(), PRED.ravel())
    average_precision["micro"] = average_precision_score(GT, PRED, average='micro')
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #  .format(average_precision["micro"]))


    # Plot all PR curves and save

    # create iso-f1 curves and plot on top the PR curves for every class
    fig, ax = plt.subplots(figsize=(10,10))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for idx, f_score in enumerate(f_scores):
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        if idx == 0:
            l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2, label='iso-f1 curves')
        else:
            l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    # labels.append('iso-f1 curves')
    l, = ax.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
                        label='micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))
    lines.append(l)
    # labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

    colors = cycle(['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal'])
    for i, color in zip(range(n_classes), colors):
        ax.plot(recall[i], precision[i], color=color, lw=lw,
                label='Precision-recall curve of class {:9s} (area = {:0.2f})'
                ''.format(classes[i], average_precision[i]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('True positive rate - Recall [TP/(TP+FN)]', fontsize=20)
    ax.set_ylabel('Positive predicted value - Precision [TP/(TP+TN)]', fontsize=20)
    ax.set_title('Multi-class Precision-recall curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=12)

    # save is needed
    if savePath is not None:
        # set up name
        if saveName is None:
            saveName = "Multiclass_PR"

        if os.path.isdir(savePath):
            fig.savefig(os.path.join(savePath, f'{saveName}.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(savePath, f'{saveName}.png'), bbox_inches='tight', dpi = 100)
        else:
            raise ValueError('Invalid save path: {}'.format(os.path.join(savePath, f'{saveName}')))

    if draw is True:
        plt.draw()
    else:
        plt.close()

    return precision, recall, average_precision, F1


## OTHER PLOTTINGS

# Helper function to show a batch
def show_batch_2D(sample_batched, title=None, img_per_row=10):

    from mpl_toolkits.axes_grid1 import ImageGrid
    '''
    Creates a grid of images with the samples contained in a batch of data.

    INPUT
    sample_batch : tuple
        COntains the actuall images (sample_batch[0]) and their label
        (sample_batch[1]).
    title : str
        Title of the created image
    img_per_row : int
        number of images per row in the created grid of images.
    '''
    batch_size = len(sample_batched[0])
    nrows = batch_size//img_per_row
    if batch_size%img_per_row > 0:
        nrows += 1
    ncols = img_per_row

    # make figure grid
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,
                    nrows_ncols=(nrows, ncols),
                    axes_pad=0.3,  # pad between axes in inch.
                    label_mode='L',
                    )

    # fill in the axis
    for i in range(batch_size):
        img = np.squeeze(sample_batched[0][i,:,:])
        grid[i].imshow(img, cmap='gray', interpolation=None)
        grid[i].set_xticks([])
        grid[i].set_yticks([])
        grid[i].set_title(sample_batched[1][i])
    if title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle('Batch of data', fontsize=20)
    plt.show()


# Helper function to show a batch
def show_batch_2D_with_histogram(sample_batched, title=None):

    from mpl_toolkits.axes_grid1 import ImageGrid
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    """
    Creates a grid of images with the samples contained in a batch of data.
    Here showing 5 random examples along with theirplt histogram.

    Parameters
    ----------
    sample_batch : tuple
        COntains the actuall images (sample_batch[0]) and their label
        (sample_batch[1]).
    title : str
        Title of the created image
    img_per_row : int
        number of images per row in the created grid of images.
    """
    n_images_to_show = 5
    index_samples = random.sample(range(len(sample_batched[0])), n_images_to_show)

    # make figure grid
    fig , ax = plt.subplots(nrows=2, ncols=n_images_to_show, figsize=(15,10))

    # fill in the axis with the images and histograms
    for i, img_idx in zip(range(n_images_to_show),index_samples) :
        img = np.squeeze(sample_batched[0][img_idx,:,:])
        ax[0][i].imshow(img, cmap='gray', interpolation=None)
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        ax[0][i].set_title(sample_batched[1][img_idx])

        # add histogram
        ax[1][i].hist(img.flatten(), bins=256)
        ax[1][i].set_xlim([-1.1,1.1])
    if title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle('Batch of data', fontsize=20)

    plt.show()

## TIME

def tictoc(tic=0, toc=1):
    '''
    # Returns a string that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    '''
    elapsed = toc-tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem*1000

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds)

def tictoc_from_time(elapsed=1):
    '''
    # Returns a string that contains the number of days, hours, minutes and
    seconds given the elapsed time
    '''
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem*1000

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds)

##
'''
Grad-CAM implementation [1] as described in post available at [2].

[1] Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-cam:
    Visual explanations from deep networks via gradient-based localization.
    InProceedings of the IEEE international conference on computer vision 2017
    (pp. 618-626).

[2] https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

'''

class gradCAM:
    def __init__(self, model, classIdx, layerName=None, use_image_prediction=True, ViT=False, debug=False):
        '''
        model: model to inspect
        classIdx: index of the class to ispect
        layerName: which layer to visualize
        '''
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        self.debug = debug
        self.use_image_prediction = use_image_prediction
        self.is_ViT = ViT

        # if the layerName is not provided, find the last conv layer in the model
        if self.layerName is None:
            self.layerName = self.find_target_layer()
        else:
            if self.debug is True:
                print('GradCAM - using layer {}'.format(self.model.get_layer(self.layerName).name))

    def find_target_layer(self):
        '''
        Finds the last convolutional layer in the model by looping throught the
        available layers
        '''
        for layer in reversed(self.model.layers):
            # check if it is a 2D conv layer (which means that needs to have
            # 4 dimensions [batch, width, hight, channels])
            if len(layer.output_shape) == 4:
                # check that is a conv layer
                if layer.name.find('conv') != -1:
                    if self.debug is True:
                        print('GradCAM - using layer {}'.format(layer.name))
                    return layer.name

        if self.layerName is None:
            # if no convolutional layer have been found, rase an error since
            # Grad-CAM can not work
            raise ValueError('Could not find a 4D layer. Cannot apply GradCAM')

    def compute_heatmap(self, image, eps=1e-6):
        '''
        Compute the L_grad-cam^c as defined in the original article, that is the
        weighted sum over feature maps in the given layer with weights based on
        the importance of the feature map on the classsification on the inspected
        class.

        This is done by supplying
        1 - an input to the pre-trained model
        2 - the output of the selected conv layer
        3 - the final softmax activation of the model
        '''
        # this is a gradient model that we will use to obtain the gradients from
        # with respect to an image to construct the heatmaps
        gradModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])

        # replacing softmax with linear activation
        gradModel.layers[-1].activation = tf.keras.activations.linear

        if self.debug is True:
            gradModel.summary()

        # use the tensorflow gradient tape to store the gradients
        with tf.GradientTape() as tape:
            '''
            cast the image tensor to a float-32 data type, pass the
            image through the gradient model, and grab the loss
            associated with the specific class index.
            '''
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            # check if the prediction is a list (VAE)
            if type(predictions) is list:
                # the model is a VEA, taking only the prediction
                predictions = predictions[4]
            pred = tf.argmax(predictions, axis=1)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        # sometimes grads becomes NoneType
        if grads is None:
            grads = tf.zeros_like(convOutputs)
        '''
        compute the guided gradients.
         - positive gradients if the classIdx matches the prediction (I want to
            know which values make the probability of that class to be high)
         - negative gradients if the classIdx != the predicted class (I want to
            know which gradients pushed down the probability for that class)
        '''
        if self.use_image_prediction == True:
            if self.classIdx == pred:
                castConvOutputs = tf.cast(convOutputs > 0, tf.float32)
                castGrads = tf.cast(grads > 0, tf.float32)
            else:
                castConvOutputs = tf.cast(convOutputs <= 0, tf.float32)
                castGrads = tf.cast(grads <= 0, tf.float32)
        else:
            castConvOutputs = tf.cast(convOutputs > 0, tf.float32)
            castGrads = tf.cast(grads > 0, tf.float32)
        guidedGrads = castConvOutputs * castGrads * grads

        # remove teh batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the weight value for each feature map in the conv layer based
        # on the guided gradient
        weights = tf.reduce_mean(guidedGrads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # now that we have the astivation map for the specific layer, we need
        # to resize it to be the same as the input image
        if self.is_ViT:
            dim = int(np.sqrt(cam.shape[0]))
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cam.numpy().reshape((dim, dim))
            heatmap = cv2.resize(heatmap,(w, h))
        else:
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(),(w, h))

        # normalize teh heat map in [0,1] and rescale to [0, 255]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = (numer/denom)
        heatmap_raw = (heatmap * 255).astype('uint8')

        # create heatmap based ont he colormap setting
        heatmap_rgb = cv2.applyColorMap(heatmap_raw, cv2.COLORMAP_VIRIDIS).astype('float32')

        return heatmap_raw, heatmap_rgb

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):

        # create heatmap based ont he colormap setting
        heatmap = cv2.applyColorMap(heatmap, colormap).astype('float32')

        if image.shape[-1] == 1:
            # convert image from grayscale to RGB
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB).astype('float32')

        output = cv2.addWeighted(image, alpha, heatmap, (1 - alpha), 0)

        # return both the heatmap and the overlayed output
        return (heatmap, output)



