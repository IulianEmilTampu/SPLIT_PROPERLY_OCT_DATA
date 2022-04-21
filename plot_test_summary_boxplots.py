'''
Script that given the summary of the models on the test dataset saved in a
tabular form (see test_model.py for more information regarding the format of
the csv file), saves plots the model performance of all the models and classes
at the same time.
'''

import os
import glob
import time
import csv
import json
import platform
import numpy as np
import pandas as pd
import seaborn as sns
import pickle5 as pickle
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle


## utilities

def add_median_labels(ax, precision='.1f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{precision}}', ha='center', va='center',
                       fontweight='bold', color='black')
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=2, foreground=median.get_color()),
        #     path_effects.Normal(),
        # ])
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),
            path_effects.Normal(),
        ])

def add_ensemble_values(df_5folds, df_ensemble, ax, hue_order, metric_to_plot):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    # define markers to use
    available_markers = ['s', 'v', '^', 'p','X', '8', '*']
    markers = cycle([ h for h in available_markers[0:len(hue_order)]])

    # get list of unique classifications
    unique_classifications = df_5folds.classification_type.unique()
    # make some tricks to be able to index the df_ensamble later
    unique_classifications = [x for x in unique_classifications for _ in range(len(hue_order))]
    models = cycle(hue_order)

    # loop through the different boxes
    for idx, median in enumerate(lines[4:len(lines):lines_per_box]):
        # get x location of the box
        x = median.get_data()[0].mean()
        # get y location based on the value of the ensemble
        # # get which model we are looking at
        m = next(models)
        y = float(df_ensemble.loc[(df_ensemble['classification_type'] == unique_classifications[idx]) & (df_ensemble['model_type'] == m)][metric_to_plot])
        ax.scatter(x, y,
                    marker=next(markers),
                    color='k',
                    edgecolors='white',
                    s=150,
                    zorder=5)


# # # shaddow in the background
def add_shadow_between_hues(ax, y_min, y_max, alpha=0.05, zorder=30, color='black'):
    # get hue region positions
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))

    # get the number of boxes per hue region
    nbr_boxes_per_hue = int(len(boxes) / len(ax.get_xticks()))

    # build coordinate regions where to add shadow
    # starting from the 0th or 1st hue gerion
    start_hue_region = 0
    # take the initial coordinate of the first box of the region and
    # the last of the last box of the region
    # get coordinatex for all boxes in order
    x_boxes_coordinates = []
    for idx, median in enumerate(lines[4:len(lines):lines_per_box]):
        # get x location of the box
        x_boxes_coordinates.append(median.get_data()[0])

    # get hue region coordinate
    hue_region_coordinatex = []
    for hue_region in range(len(ax.get_xticks())):
        idx_first_box = hue_region*nbr_boxes_per_hue
        idx_last_box = idx_first_box + nbr_boxes_per_hue -1
        hue_region_coordinatex.append([x_boxes_coordinates[idx_first_box][0], x_boxes_coordinates[idx_last_box][-1]])

    # loop through the regions and color
    for c in range(start_hue_region, len(ax.get_xticks()),2):
        x_min, x_max = hue_region_coordinatex[c][0], hue_region_coordinatex[c][-1]
        ax.add_patch(Rectangle((x_min, y_min), (x_max - x_min), (y_max-y_min), color=color, alpha=alpha, zorder=zorder))

## paths

BASE_FOLDER = '/flush/iulta54/Research/P3-OCT_CLASSIFICATION_summary_trained_models/RETINAL-AIIMS'

# SUMMARY_FILE_PATH = '/flush/iulta54/Research/P3-OCT_CLASSIFICATION_summary_trained_models/THR/overall_tabular_test_summary.csv'
SUMMARY_FILE_PATH = os.path.join(BASE_FOLDER,'overall_tabular_test_summary_best_performing_model_versions.csv')
DF = pd.read_csv(SUMMARY_FILE_PATH)

## load csv file and put in dataframe

# print(DF.head())

# ## SIMPLE BOX PLOT - filter dataframe and plot
# save_images = False
# model_version = 'last'
# # df = DF.loc[DF['model_version'] == model_version]
# df = DF
#
# tick_font_size=10
# title_font_size=20
# label_font_size=20
# legend_font_size = 15
#
# fig, box_plot = plt.subplots(nrows=1, ncols=0, figsize=(15,15))
#
# box_plot = sns.boxplot(x="classification_type",
#                        y="accuracy",
#                        hue='model_type',
#                        hue_order = ['LightOCT', 'ResNet50', 'M4', 'M6', 'ViT'],
#                        data=df,
#                        palette="Set3",
#                        )
# box_plot.set_title('Classification summary thyroid data', fontsize=title_font_size)
# box_plot.tick_params(labelsize=tick_font_size)
# box_plot.set_ylabel('Accuracy', fontsize=label_font_size)
# # format y tick labels
# ylabels = [f'{x:,.2f}' for x in box_plot.get_yticks()]
# box_plot.set_yticklabels(ylabels)
#
# box_plot.set_xlabel('Input configuration', fontsize=0)
# box_plot.set_xticklabels(box_plot.get_xticklabels(),rotation=0)
# # plt.setp(box_plot.get_legend().get_texts(), fontsize=legend_font_size)
#
# box_plot.yaxis.grid(True) # Hide the horizontal gridlines
# box_plot.xaxis.grid(False) # Show the vertical gridlines
#
# # add pattern to boxplots and legend
# available_hatches = ['////', '\\\\', 'xx', 'oo','.', '/', '\\', '|', '-',]
# hatches = cycle([ h for h in available_hatches[0:len(df.model_type.unique())]])
# colors, legend_hatch = [], []
#
# for i, patch in enumerate(box_plot.artists):
#     # Boxes from left to right
#     hatch = next(hatches)
#     patch.set_hatch(hatch)
#     colors.append(patch.get_facecolor())
#     legend_hatch.append(hatch)
#
# # fix legend
# labels = [l.get_text() for l in box_plot.legend_.texts]
# colors = colors[0:len(df.model_type.unique())]
# legend_hatch = legend_hatch[0:len(df.model_type.unique())]
# legend_handles = [mpatches.Patch(facecolor=c, hatch=h,label=l) for c,h,l in zip(colors, legend_hatch, labels)]
# box_plot.legend(loc='best', handles = legend_handles)
#
# # # add median values on boxplots
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.patheffects as path_effects
#
# add_median_labels(box_plot)
#
# box_plot.figure.tight_layout()
#
# if save_images == True:
#     file_name = f'overall_classification_summary_optimal_models'
#     fig.savefig(os.path.join(BASE_FOLDER, file_name +'.pdf'), bbox_inches='tight', dpi = 100)
#     fig.savefig(os.path.join(BASE_FOLDER, file_name +'.png'), bbox_inches='tight', dpi = 100)
# else:
#     plt.show()


## BOXPLOT OF BEST MODELS WITH ENSEMBLE OVERLAYED
save_images = True
plt.rcParams["font.family"] = "Times New Roman"

# model_version = 'last'
metrics = ['precision', 'recall', 'accuracy', 'f1-score', 'auc', 'matthews_correlation_coef']
metric_name_for_plot = ['Precision [0,1]', 'Recall [0,1]', 'Accuracy [0,1]', 'F1-score [0,1]', 'AUC [0,1]', 'Matthews correlation coefficient [-1,1]']
# metrics = ['precision']
for metric_to_plot in metrics:

    hue_order = ['LightOCT', 'ResNet50', 'M4', 'M6', 'ViT']

    df_5folds = DF.loc[(DF['model_version'] == 'best') | (DF['model_version'] == 'last')]
    df_ensemble = DF.loc[DF['model_version'] == 'ensemble']

    unique_classificatios = df_5folds.classification_type.unique()

    tick_font_size=17
    title_font_size=20
    label_font_size=20
    legend_font_size = 18


    fig, box_plot = plt.subplots(nrows=1, ncols=0, figsize=(15,10))

    box_plot = sns.boxplot(x="dataset",
                        y=metric_to_plot,
                        hue='model_type',
                        hue_order = hue_order,
                        data=df_5folds,
                        palette="Set3",
                        )

    box_plot.set_title(f'Classification performance', fontsize=title_font_size)
    box_plot.tick_params(labelsize=tick_font_size)
    box_plot.set_ylabel(f'{metric_name_for_plot[metrics.index(metric_to_plot)]}', fontsize=label_font_size)
    # format y tick labels
    ylabels = [f'{x:,.2f}' for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels)

    box_plot.set_xlabel('Input configuration', fontsize=0)
    box_plot.set_xticklabels(box_plot.get_xticklabels(),rotation=0)
    # plt.setp(box_plot.get_legend().get_texts(), fontsize=legend_font_size)

    # add pattern to boxplots and legend
    available_hatches = ['////', '\\\\', 'xx', 'oo','.', '/', '\\', '|', '-',]
    hatches = cycle([ h for h in available_hatches[0:len(df_5folds.model_type.unique())]])
    colors, legend_hatch = [], []

    # fix legend
    # # boxplots
    for i, patch in enumerate(box_plot.artists):
        # Boxes from left to right
        hatch = next(hatches)
        patch.set_hatch(hatch)
        colors.append(patch.get_facecolor())
        legend_hatch.append(hatch)

    labels = [f'{l.get_text()} (5 folds)' for l in box_plot.legend_.texts]
    colors = colors[0:len(hue_order)]
    legend_hatch = legend_hatch[0:len(hue_order)]
    legend_handles = [mpatches.Patch(facecolor=c, hatch=h,label=l) for c,h,l in zip(colors, legend_hatch, labels)]

    # # ensamble models
    available_markers = ['s', 'v', '^', 'p','X', '8', '*']
    markers = cycle([ h for h in available_markers[0:len(hue_order)]])
    lables = [f'{m} ensemble' for m in hue_order]
    [legend_handles.append(mlines.Line2D([], [], color='k', marker=next(markers), linestyle='None',
                            markersize=10, label=l)) for l in lables]

    box_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), handles = legend_handles, fontsize=legend_font_size, ncol=2)

    add_ensemble_values(df_5folds, df_ensemble, box_plot, hue_order, metric_to_plot)

    # add shadows
    y_min = box_plot.get_ylim()[0]
    y_max = box_plot.get_ylim()[-1]
    add_shadow_between_hues(box_plot, y_min, y_max, alpha=0.01, zorder=60, color='blue')


    box_plot.yaxis.grid(True, zorder=-3) # Hide the horizontal gridlines
    box_plot.xaxis.grid(False) # Show the vertical gridlines
    box_plot.figure.tight_layout()

    if save_images == True:
        file_name = f'overall_classification_summary_optimal_models_{metric_to_plot}_Kermany_AIIMS'
        fig.savefig(os.path.join(BASE_FOLDER, file_name +'.pdf'), bbox_inches='tight', dpi = 100)
        fig.savefig(os.path.join(BASE_FOLDER, file_name +'.png'), bbox_inches='tight', dpi = 100)
        plt.close(fig)
    else:
        plt.show()