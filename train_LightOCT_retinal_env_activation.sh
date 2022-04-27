#!/bin/bash

Help()
{
   # Display Help
   echo "Bash script to run multiple trainings"
   echo
   echo "Syntax: run_training [w|d|g]"
   echo "required inputs:"
   echo "w     Working folder (where the scripts are)"
   echo "d     Dataset folder (were the data is located)"
   echo "g     GPU number on which to run training"
   echo
}

while getopts w:hd:g: option; do
case "${option}" in
   h) # display Help
       Help
       exit;;
   w) working_folder=${OPTARG};;
   d) dataset_folder=${OPTARG};;
   g) gpu=${OPTARG};;

   \?) # incorrect option
         echo "Error: Invalid input"
         exit 1
esac
done


# make sure to have the right conda environment open when running the script
# activate conda environment
eval "$(conda shell.bash hook)"
conda activate P5

# work on GPU 0
export CUDA_VISIBLE_DEVICES=$gpu

# go to the working folder
cd $working_folder

# create trained_log_file folder
if ! [ -d $working_folder/trained_models_log ]; then
   echo "Creating folder to save log."
   mkdir $working_folder/trained_models_log
fi

log_folder=$working_folder/trained_models_log


 ############################################################################
 ################################# TRAINING  ################################
 ############################################################################

model_configuration=LightOCT

# declare modelspecification variable
declare -a model_configuration=LightOCT
declare -a batchSize=64
declare -a lr=0.0001
declare -a loss=cce

# declare dataset specification
declare -a dataset_type=retinal
declare -a dataset_split_strategy=per_image
declare -a ids=none
declare -a n_folds=3
declare -a n_rkf=1

# experiment specifications
declare -a random_label_experiment=True


save_model_name="$model_configuration"_"$dataset_split_strategy"_split_"$n_folds"_folds_rkf_"$n_rkf"_lr"$lr"_batch_"$batchSize"_"$dataset_type"_rls_"$random_label_experiment"
python3 -u configure_training.py -wd $working_folder -df $dataset_folder -dt $dataset_type -dss $dataset_split_strategy -mc $model_configuration -mn $save_model_name -b $batchSize -f $n_folds -nkf $n_rkf -l $loss -lr $lr -ids $ids -v 2 -db False -rle $random_label_experiment |& tee $log_folder/$save_model_name.log

python3 -u run_training.py -cf $working_folder/trained_models/$save_model_name/config.json -e 250 -p 250 -db False |& tee -a $log_folder/$save_model_name.log

 ############################################################################
 ################################# TESTING  #################################
 ############################################################################
python3 -u test_model.py -m $working_folder/trained_models/$save_model_name -d $dataset_folder -mv best |& tee -a $log_folder/$save_model_name.log
python3 -u test_model.py -m $working_folder/trained_models/$save_model_name -d $dataset_folder -mv last |& tee -a $log_folder/$save_model_name.log

