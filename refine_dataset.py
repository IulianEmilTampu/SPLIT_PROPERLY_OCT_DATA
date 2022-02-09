'''
Script to re-organize and splits per class Kermany's dataset doi 10.17632/rscbjbr9sj.2
https://data.mendeley.com/datasets/rscbjbr9sj/2.

This script can also create training, test and validation sets for a one fold training
using a per-volume split trategy.
NOTE that this is not required by the other scripts since split is taken care
internally. However, if one wants to have a version of the dataset with a proper split,
one can have it.

Steps
1 - from the train and test take all the images and save per class organization.
2 - Since there are images from the same patient in multiple sets, keep track
of the name changes if there is any collision.
3 - make proper split, with test images from each class coming from subjects
    not in the training or validation sets. Use the same number of testing and
    validation images as the original dataset (test - 242 for each class,
    val - 8 for each class)
'''

import os
import glob
import pathlib
import shutil
import csv
import numpy as np
import random

## utility functions
def get_id(file):
    '''
    Returns the id of the file. The base name of the file expected to be named as:
    Class-ID-numeric_attribute.jpeg
    '''
    id_delimiters = [os.path.basename(file).find("-")+1,  os.path.basename(file).rfind("-")]
    return os.path.basename(file)[id_delimiters[0]:id_delimiters[1]]

def get_per_id_files(files):
    '''
    Utility that given a list of files returns a list of the unique identifiers
    and the files organized per-identifier
    '''
    # get all unique identifiers
    unique_IDs = [get_id(file) for file in files]
    unique_IDs = list(dict.fromkeys((unique_IDs)))

    # now get the files organized per id
    per_id_organization = []
    for ID in unique_IDs:
        per_id_organization.append([ID,[file for file in files if f'-{ID}-' in file]])

    return per_id_organization

## get all file names form the original train, test and val splits

dataset_folder = 'path/to/the/original/dataset/where/the/train_test/folders/are'
destination_folder = "path/to/destination/folder"

if not os.path.isdir(destination_folder):
    print(f'The per class files will be ssaved at {destination_folder}')
    os.mkdir(destination_folder)

train_folder = os.path.join(dataset_folder, 'train')
test_folder = os.path.join(dataset_folder, 'test')

classes = [os.path.basename(f) for f in glob.glob(os.path.join(test_folder, '*'))]

# variable that controlls if a per-volume dataset should be created
create_per_volume_split_dataset = False

## copy the test dataset classes in the destination folder

print('Copying training dataset...')
[shutil.copytree(os.path.join(train_folder, c), os.path.join(destination_folder, c)) for c in classes]

## create csv file where to save the information about the renamed files

summary_file = os.path.join(destination_folder, "renamed_during_conversion.csv")
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_rows = []
csv_header = ['identifier','original_split', 'class', 'original_file_name', 'new_file_name']

writer.writerow(csv_header)

## 'copy' test files, renaming them if necessary
# use a buffer for copying so that we transfer the files once and not multiple
# times. Useful for debugging
buffer_to_copy = []

for c in classes:
    files_to_copy = glob.glob(os.path.join(test_folder, f'{c}/*.jpeg'))
    for idx, f in enumerate(files_to_copy):
        print(f'Buffering test files for copying ({idx:4d}/{len(files_to_copy):4d})\r', end='')
        # check if file is already in the destination folder
        if os.path.isfile(os.path.join(destination_folder, c, os.path.basename(f))):
            ''' File already exists so need to renamed it.
            1 - Get all the files in the destination forlder sharing the
                same identifier.
            2 - Order them and get the numerical attribute of the last file
            3 - Save file by using the next available numerical attribute.
            '''
            id_delimiters = [os.path.basename(f).find("-")+1,  os.path.basename(f).rfind("-")]
            ID = os.path.basename(f)[id_delimiters[0]:id_delimiters[1]]


            files_with_same_id = [os.path.basename(i) for i in glob.glob(os.path.join(destination_folder, f'{c}/*.jpeg')) if f'-{ID}-' in i]
            file_with_largest_numerical_attribute = files_with_same_id[np.argmax(np.array([int(i[id_delimiters[1]+1:i.find(".jpeg")]) for i in files_with_same_id]))]
            old_numerical_attribute = os.path.basename(f)[id_delimiters[1]+1:os.path.basename(f).find(".jpeg")]
            new_numerical_attribute = int(file_with_largest_numerical_attribute[id_delimiters[1]+1:file_with_largest_numerical_attribute.find(".jpeg")])
            new_name = os.path.basename(f).replace(f'-{old_numerical_attribute}.', f'-{new_numerical_attribute+1}.')

            # save information about the name change
            row = [ID, 'test', c, os.path.basename(f), new_name]
            writer.writerow(row)

            # save information for later copying
            buffer_to_copy.append([f, os.path.join(destination_folder, c, new_name)])

        else:
            # save information for later copying
            buffer_to_copy.append([f, os.path.join(destination_folder, c, os.path.basename(f))])

## actually copy the files

for idx, f in enumerate(buffer_to_copy):
    shutil.copy(f[0], f[1])
    print(f'Copied {idx+1:4d} out of {len(buffer_to_copy):4d} \r', end='')

csv_file.close()

## Work on splitting the data
create_per_volume_split_dataset = True
if create_per_volume_split_dataset:
    per_volume_split_destination_folder = '/flush/iulta54/Research/Data/OCT/Retinal/Zhang_dataset_version_3/per_volume_split'


    dataset_dict = {
                    "test":{
                            classes[0]:[],
                            classes[1]:[],
                            classes[2]:[],
                            classes[3]:[],
                            },
                    "train":{
                            classes[0]:[],
                            classes[1]:[],
                            classes[2]:[],
                            classes[3]:[],
                            },
                    "val":{
                            classes[0]:[],
                            classes[1]:[],
                            classes[2]:[],
                            classes[3]:[],
                            }
                        }

    # setting the number of images to select for every class and the number of
    # and the number of minimum IDs from where these images are taken

    n_test_per_class = 242
    n_val_per_class = 8
    min_n_IDs = 2

    for c in classes:
        print(f'Working on class {c}...')
        # get all the files for this class
        original_files = glob.glob(os.path.join(destination_folder, f'{c}/*.jpeg'))
        # get files splitted per IDs
        per_id_organized = get_per_id_files(original_files)
        # shuffle list
        random.shuffle(per_id_organized)
        print(f'    Shuffling IDs...')

        # ############## TEST SET
        # take out as many samples as needed to get the test dataset
        print(f'    Making test set... \r', end='')
        aus_files = []
        test_selected_files_index = []
        used_IDs = []
        ID_counter = 0
        while (len(aus_files) < n_test_per_class) or (ID_counter <= min_n_IDs):
            used_IDs.extend([per_id_organized[ID_counter][0]])
            aus_files.extend(per_id_organized[ID_counter][1])
            test_selected_files_index.extend([original_files.index(i) for i in per_id_organized[ID_counter][1]])

            ID_counter += 1

        # now that I have a bag of files from n different IDs, randomly take as many
        # as needed
        selected_files = random.sample(aus_files, n_test_per_class)

        # save information in the dataset dictionary
        dataset_dict["test"][c]=selected_files

        print(f'    Test set done! Using {n_test_per_class} from {len(used_IDs)} different IDs')

        # ############## VAL SET
        print(f'    Making validation set... \r', end='')
        # take out as many samples as needed to get the test dataset
        aus_files = []
        validation_selected_files_index = []
        used_IDs = []
        # continue using the same ID_counter since the IDs were shuffled at the beginning
        last_ID_counter = ID_counter
        while (len(aus_files) < n_val_per_class) or (ID_counter <= min_n_IDs+last_ID_counter):
            used_IDs.extend([per_id_organized[ID_counter][0]])
            aus_files.extend(per_id_organized[ID_counter][1])
            validation_selected_files_index.extend([original_files.index(i) for i in per_id_organized[ID_counter][1]])

            ID_counter += 1

        # now that I have a bag of files from n different IDs, randomly take as many
        # as needed
        selected_files = random.sample(aus_files, n_val_per_class)

        # save information in the dataset dictionary
        dataset_dict["val"][c]=selected_files

        print(f'    Validation set done! Using {n_val_per_class} from {len(used_IDs)} different IDs')

        # ############## TRAINING SET
        all_selected_file_indexes = test_selected_files_index + validation_selected_files_index
        selected_files = [file for idx, file in enumerate(original_files) if idx not in all_selected_file_indexes]
        print(f'    Trainin set done! Using {len(selected_files)}')
        # save information in the dataset dictionary
        dataset_dict["train"][c]=selected_files

    ## Copy files in the right directories
    # make destination folder
    [pathlib.Path(os.path.join(per_volume_split_destination_folder, v)).mkdir(parents=True, exist_ok=True) for v in ["train", "test", "val"]]

    for split, per_class_files in dataset_dict.items():
        print(f'Copying {split} files...')
        for c, file_list in per_class_files.items():
            save_path = os.path.join(per_volume_split_destination_folder, split, c)
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

            for idx, file in enumerate(file_list):
                print(f'    {c} -> {idx+1:5d}/{len(file_list):5d} \r', end='')
                # copy file in the right folder
                shutil.copy(file, os.path.join(save_path, os.path.basename(file)))
        print('Done! \n')




