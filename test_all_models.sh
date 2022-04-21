#!/bin/bash

declare -a working_directory="/flush/iulta54/Research/P3_OCT_SPLIT_PROPERLY_YOUR_DATA"
cd $working_directory
declare -a dataset_folder="/flush/iulta54/Research/Data/OCT/AIIMS_Dataset/original"
declare -a models_base_folder="/flush/iulta54/Research/P3-OCT_CLASSIFICATION_summary_trained_models/BT"

# for model in $models_base_folder/5_folds/*/ ; do
#     echo $model
#     python3 test_model.py -m $model -d $dataset_folder -mv best
#     python3 test_model.py -m $model -d $dataset_folder -mv last
# done


# aggregate all the results in one .csv file
python3 aggregate_tabular_test_summary.py -m $models_base_folder

