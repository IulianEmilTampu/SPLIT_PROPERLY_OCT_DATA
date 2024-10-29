# Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images

This repository contains the code used to train a shallow CNN model (LightOCT) [1] for classification on two publicly available optical coherence tomography (OCT) datasets using a *per-image* or *per-volume/subject* dataset splitting strategy. The results that can be obtained with this code are representative of how improper dataset splitting can inflate model accuracy performance.

[Journal](https://doi.org/10.1038/s41597-022-01618-6) | [Cite](#reference)

**Abstract**


**Key highlights:**
- **Data leakage** between training and test sets can originate when using 2D images extracted from 3D data and disregarding the relation that exists between images from the same volume.  
- **Improper dataset splitting** can inflate test model performance by up to 30% on experiments using OCT data.
- **Importance**: Highlights the need for careful dataset handling to ensure accurate and reliable model evaluation.

## Table of Contents
- [Prerequisites and setup](#setup)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Reference](#reference)
- [License](#license)

### Prerequisites and setup
A computer with GPU and relative drivers installed.
Anaconda or Miniconda Python environment manager.

## Setup
1. Clone the repo to your local machine
```sh
git clone git@github.com:IulianEmilTampu/OCT_SPLIT_PROPERLY_YOUR_DATA.git
```
2. Move to the downloaded repository and create a Python environment using the given .yml file
  ```sh
   conda env create -f environment_setup.yml
   ```

## Datasets
### AIIMS dataset
The AIIMS dataset is available at https://www.bioailab.org/datasets [2]. 
This dataset is ready to use since the images are saved per class (Healthy or Cancer) and per subject.

### Kermany's retinal dataset
Kermany's retinal dataset is available at https://data.mendeley.com/datasets/rscbjbr9sj [3, 4]
The dataset comes with the train and test splits already organized. This version of the dataset is called *original dataset*. To obtain a version of the dataset split per class, use the **refine_dataset.py** script, where the location of the original dataset can be specified using the variable dataset_folder and the location of where to save the reorganized dataset by destination_folder. These two variables are specified within the refine_dataset.py 

### Srinivasan’s retinal dataset
Srinivasan’s retinal dataset is available at https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.html.
This dataset is ready to use since the images are saved per class (age-related macular degeneration (AMD), diabetic macular edema (DME), and normal subjects) and per subject. 

## Code structure
The repository is organized as follows:

- **Data Preparation**:
  - `refine_dataset.py`: Script to re-organize and splits per class Kermany's dataset. 
- **Model Training and Evaluation**:
  - `configure_training.py`: Script generates the configuration file used to run model training.
  - `run_training.py`: Scripts for training the LightOCT models.
  - `test_model.py`: Script to evaluate models on test data.
- **Utilities and Visualization**:
  - `models_tf.py`: Contains the models architecture definition.
  - `utilities.py` and `utilities_models_tf.py`: Utility functions for data handling, model evaluation, and performance metrics.
  - `visualize_dataset.py`: Tools for dataset inspection.
  - `aggregate_tabular_test_summary.py`: Script that aggregates all the tabular test summary for all the trained models.

## Usage
### UNIX-based operating systems
The repository comes with two shell scripts that run automatically model configuration, training and testing for the retinal and the AIIMS datasets. By default, the training, validation, and testing datasets are created using a *per-volume/subject* splitting strategy, and models are trained through a 3-fold cross-validation scheme. To change the splitting strategy, open the **train_LightOCT_retinal.sh** or the **train_LightOCT_AIIMS.sh** files and change the dataset_split_strategy variable (line 60) to *per-image* or *original* (*original* only valid for the retinal dataset).

To run the default training on the retinal dataset, activate the Python environment created using the instructions above and, in the folder where the repository is copied, run
  ```sh
  ./train_LightOCT_retinal.sh -w path/to/where/the/repository/is/located -d /path/to/the/per_class/retinal/dataset -g 0
  ```
This will:
1. Evoke the **configure_training.py** script that takes, among others, the path to the folder where the different scripts are located (whereafter working directory, parameter **-w**) and then to the folder where the dataset is located (parameter **-d**). The last parameter (**-g**) specifies on which GPU the code will be run. Note that the dataset path has to match the type of splitting strategy that is set. In the case of dataset_split_strategy=*original*, the dataset path should point to the dataset as it was downloaded. In the case dataset_split_strategy=*per-image* or *per-volume*, the path should point to the dataset organized as per-class (the one obtained using the **refine_dataset.py** script).\
The **configure_training.py** splits the dataset using the specified strategy, sets the model parameters based on the dataset configuration, and saves the configuration information in a **config.json** file. The **config.json** file is saved in the working directory under *trained_models/Model_name*
2. Run the **run_training.py** script that, using the **config.json** file, trains the LightOCT model.
6. Test both the best and the last models trained using early stopping (best model = model with the highest accuracy, last model = the actual last model before early stopping stopped the training). The test is run by invoking the **test_model.py** script.

Models and training performances are saved independently for every fold in a folder dedicated for the model just trained (*trained_models/Model_name*). Test results are also saved in the dedicated folder as ROC and PP curves, as well as the raw model predictions for the test dataset.

Use 
  ```sh
  ./train_LightOCT_AIIMS.sh -w path/to/where/the/repository/is/located -d /path/to/the/per_class/AIIMS/dataset -g 0
  ```
to run the same routine as before on the AIIMS dataset.

### WINDOWS-based operating systems
For users using Windows-based systems, the **configure_training.py**, **run_training.py** and **test_model.py** scripts can be run independently from the command line. For an understanding of the required and default parameters, run *python3 script_name.py --help*.


## Citation
If you use this work, please cite:

```bibtex
@article{tampu_2022_inflation,
  title={Inflation of test accuracy due to data leakage in deep learning-based classification of OCT images},
  author={Tampu, Iulian Emil and Eklund, Anders and Haj-Hosseini, Neda},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={580},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments
[1]Butola *et al.,* Deep learning architecture “LightOCT” for diagnostic decision support using optical coherence tomography images of biological samples. Biomedical Optics Express. 2020 Sep 1;11(9):5017-31.\
[2] Butola *et al.,* Volumetric analysis of breast cancer tissues using machine learning and swept-source optical coherence tomography. Applied optics. 2019 Feb 10;58(5):A135-41.\
[3] Kermany *et al.,* Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell. 2018 Feb 22;172(5):1122-31.\
[4] Kermany *et al.,* Large dataset of labeled optical coherence tomography (oct) and chest x-ray images. Mendeley Data. 2018;3:10-7632.


