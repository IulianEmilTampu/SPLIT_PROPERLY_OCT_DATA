'''
Main script that runs the training of a deep leaning model for tissue
classification on OCT images.

Steps
- open the config.json file
- for every fold:
    - create dataloaders for training and validation
    - create model based on specifications
    - train model based on specifications
    - test model on test dataset
- test the ensamble of all the cross-validation models if any.
- save all testing performances
'''

import os
import sys
import json
import glob
import types
import time
import random
import argparse
import importlib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow.keras.layers as layers


## parse the configuration file

parser = argparse.ArgumentParser(description='Script that runs a cross-validation training for OCT 2D Retinal disease classification. It uses the configuration file created using the configure_training.py file. Run the configuration first!')

parser.add_argument('-cf','--configuration_file' ,required=True, help='Provide the path to the configuration file generated using the configure_training.py script.')
parser.add_argument('-db','--debug' ,required=False, help='Set to True if one wants to run the training in debug mode (only 15 epochs with 10 early stop patience).', default=False)
parser.add_argument('-e','--epocs' ,required=False, help='Set the maximum number of epochs used to train the model Default 200.', default=200)
parser.add_argument('-p','--patience' ,required=False, help='Set the patiencs for early stopping. Default 25', default=25)
args = parser.parse_args()

configuration_file = args.configuration_file
debug = args.debug == "True"
max_epochs = int(args.epocs)
patience = int(args.patience)


# # # # # # # # DEBUG
# configuration_file = '/flush/iulta54/Research/P3_1-OCT_DATASET_STUDY/trained_models/TEST/config.json'
# debug = True
# max_epochs = 2
# patience = 5

if not os.path.isfile(configuration_file):
    raise ValueError(f'Configuration file not found. Run the configure_training.py script first. Given {configuration_file}')

if debug is True:
    print(f'\n{"-"*70}')
    print(f'{"Running training routine in debug mode (using 20% data and lower # of epochs)":^20}')
    print(f'{"-"*70}\n')

    # reducing the number of training epochs
    max_epochs = 4
    patience = 4

else:
    print(f'{"-"*24}')
    print(f'{"Running training routine":^20}')
    print(f'{"-"*24}\n')

with open(configuration_file) as json_file:
    config = json.load(json_file)

# import custom scripts from the working directory
sys.path.append(config['working_folder'])
import models_tf
import utilities
import utilities_models_tf

print(f'{"Working folder":<26s}: {config["working_folder"]}')
print(f'{"Debug":<26s}: {debug}')
print(f'{"Max # epochs":<26s}: {max_epochs}')
print(f'{"Early stop patience":<26s}: {patience}')

## create folders where to save the data and models for each fold

for cv in range(config['N_FOLDS']):
    if not os.path.isdir(os.path.join(config['save_model_path'], 'fold_'+str(cv+1))):
        os.mkdir(os.path.join(config['save_model_path'], 'fold_'+str(cv+1)))

## initialise variables where to save test summary
importlib.reload(utilities_models_tf)
importlib.reload(utilities)
importlib.reload(models_tf)

test_fold_summary = {}

# ############################ TRAINING
# specify data generator specific for the different types of datasets
if config['dataset_type'] == 'retinal':
    data_gen = utilities.retinal_data_gen
elif config['dataset_type'] == 'AIIMS':
    data_gen = utilities.AIIMS_data_gen

for cv in range(config['N_FOLDS']):
    print('Working on fold {}/{}. Start time {}'.format(cv+1, config['N_FOLDS'], datetime.now().strftime("%H:%M:%S")))

    print(' - Creating datasets...')
    # create datasets
    train_dataset = data_gen(config['training'][cv][::10000],
                            unique_labels=config['unique_labels'],
                            batch_size=config['batch_size'],
                            training=True,
                            input_size=config['input_size'],
                            channels=config['n_channels'],
                            )

    val_dataset = data_gen(config['validation'][cv],
                            unique_labels=config['unique_labels'],
                            batch_size=config['batch_size'],
                            training=False,
                            input_size=config['input_size'],
                            channels=config['n_channels'])

    # create model based on specification
    if config['model_configuration'] == 'LightOCT':
        model = models_tf.LightOCT(number_of_input_channels = 1,
                        model_name=config['model_configuration'],
                        num_classes = len(config['unique_labels']),
                        data_augmentation=config['data_augmentation'],
                        class_weights = config['class_weights'],
                        kernel_size=config['kernel_size'],
                        input_size=config['input_size']
                        )

    else:
        raise ValueError(f'Specified model configuration not available. Provide one that is implemented in models_tf.py. Given {config["model_configuration"]}')
        sys.exit()

    # train model
    print(' - Training fold...')
    warm_up = False,
    warm_up_epochs = 5
    warm_up_learning_rate = 0.00001

    utilities_models_tf.train(model,
                    train_dataset, val_dataset,
                    unique_labels = config['unique_labels'],
                    loss=[config['loss']],
                    start_learning_rate = config['learning_rate'],
                    scheduler = 'constant',
                    power = 0.1,
                    max_epochs=max_epochs,
                    early_stopping=True,
                    patience=patience,
                    save_model_path=os.path.join(config['save_model_path'], 'fold_'+str(cv+1)),
                    save_model_architecture_figure=True if cv==0 else False,
                    warm_up = warm_up,
                    warm_up_epochs = warm_up_epochs,
                    warm_up_learning_rate = warm_up_learning_rate,
                    verbose=config['verbose']
                    )















