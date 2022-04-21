'''
Script that aggregates all the tabular test summary for all the trained models.
The script works on the folder where for every classification task the 5_folds
model trainings are saved.
'''

import os
import sys
import cv2
import csv
import glob
import numpy as np
import argparse

## get paths and create summary tabular file

# parser = argparse.ArgumentParser(description='Script that aggregates test model performance obtained through the test_model.py.')
# parser.add_argument('-m','--all_model_path' ,required=True, help='Path to there all the classification tasks folders are located')
# args = parser.parse_args()
#
# all_model_path = args.all_model_path

# # # # # # DEBUG
all_model_path = '/flush/iulta54/Research/P3-OCT_CLASSIFICATION_summary_trained_models/RETINAL'

summary_file = os.path.join(all_model_path,f'overall_tabular_test_summary.csv')
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = ['classification_type',
            'nbr_classes',
            'model_type',
            'model_version',
            'fold',
            'precision',
            'recall',
            'accuracy',
            'f1-score',
            'auc',
            'training_time',
            'matthews_correlation_coef'
            ]
writer.writerow(csv_header)

## loop through all the classifications and models, and save information
rows_to_write = []

classification_folder = glob.glob(os.path.join(all_model_path,'*',''))
metric_to_check = 'matthews_correlation_coef'
idx_metric = csv_header.index(metric_to_check)

flag_best_performing = {}

models = glob.glob(os.path.join(all_model_path,'5_folds','*',''))
for m in models:
    aus_metric_value_best_performing = 0
    # open best and last tabular information
    for mv in ['best', 'last']:
        # store mean value for the best performing metric
        # open csv:
        try:
            with open(os.path.join(m,f'{mv}_tabular_test_summary.csv'), newline='') as csvfile:
                aus_csv = spamreader = csv.reader(csvfile)
                aus_best_performing_model = []
                # skip header
                next(aus_csv)
                for row in aus_csv:
                    rows_to_write.append(row)
                    # get value for the metric_to_check
                    aus_best_performing_model.append(float(row[idx_metric]))

                # save this model version of the value mean metric value is higher than the previous one
                if np.mean(aus_best_performing_model) >= aus_metric_value_best_performing:
                    # update metric
                    aus_metric_value_best_performing = np.mean(aus_best_performing_model)
                    # save information
                    flag_best_performing[os.path.basename(os.path.dirname(m))] = {'model_version':mv}
        except:
            print(f'Missing {mv} tabular information for {m}.')

writer.writerows(rows_to_write)
csv_file.close()


## do the same but save only the best performing model version and the ensamble

summary_file = os.path.join(all_model_path,f'overall_tabular_test_summary_best_performing_model_versions.csv')
csv_file = open(summary_file, "w")
writer = csv.writer(csv_file)
csv_header = ['classification_type',
            'nbr_classes',
            'model_type',
            'model_version',
            'fold',
            'precision',
            'recall',
            'accuracy',
            'f1-score',
            'auc',
            'training_time',
            'matthews_correlation_coef'
            ]
writer.writerow(csv_header)

## loop through all the classifications and models, and save information
rows_to_write = []

# loop through the 5 fold models
models = glob.glob(os.path.join(all_model_path,'5_folds','*',''))
for m in models:
    # open the best performing model
    try:
        mv = flag_best_performing[os.path.basename(os.path.dirname(m))]["model_version"]
        with open(os.path.join(m,f'{mv}_tabular_test_summary.csv'), newline='') as csvfile:
            aus_csv = spamreader = csv.reader(csvfile)
            # skip header
            next(aus_csv)
            for row in aus_csv:
                rows_to_write.append(row)
    except:
        print()

writer.writerows(rows_to_write)
csv_file.close()








