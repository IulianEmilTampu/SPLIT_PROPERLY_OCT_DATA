
<!-- ABOUT THE PROJECT -->
## About The Project

Code used to train a shallow CNN model (LightOCT) [1] for classification on two publicly available optical coherence tomography (OCT) dataset using a *per-image* or *per-volume/subject* dataset splitting strategy. The results that can be obtained with this code are representative of of how impropper dataset splitting can inflate model accuracy performance.

### Prerequisites
A computer with GPU and relative drivers installed.
Anaconda or Miniconda python environment manager.

### Installation
1. Clone the repo to your local machine
```sh
git clone git@github.com:IulianEmilTampu/OCT_SPLIT_PROPERLY_YOUR_DATA.git
```
2. Move to the downloaded repository and create a python environment using the given .yml file
  ```sh
   conda env create -f environment_setup.yml
   ```
<!-- Dataset Preparation -->
## Dataset preparation
### AIIMS dataset
The AIIMS dataset is available at https://www.bioailab.org/datasets [2]. 
This dataset is ready to use since the images are saved per class (Healthy or Cancer) and per subject.

### Retinal dataset
The retinal dataset is available at https://data.mendeley.com/datasets/rscbjbr9sj [3, 4]
The dataset comes with the train and test splits already organised. This version of the dataset is called *original dataset*. To obtain a version of the dataset split per class, use the **refine_dataset.py** script, where the location of the original dataset can be specified using the variable dataset_folder and the location of where to save the reorganized dataset by destination_folder. These two variables are specified within the refine_dataset.py 

<!-- USAGE EXAMPLES -->
## Usage
### UNIX based operating systems
The repository comes with two shell scripts that run automatically model configuration, training and testing for the retinal and the AIIMS datasets. By default, the training, validation and testing datasets are created using a *per-volume/subject* splitting strategy, and models are trainied through a 3-fold cross-validation scheme. To change the splitting strategy, open the **train_LightOCT_retinal.sh** or the **train_LightOCT_AIIMS.sh** files and change the dataset_split_strategy variable (line 60) to *per-image* or *original* (*original* only valid for the retinal dataset).

To run the default training on the retinal dataset, activate the python environment created using the instructions above and, in the folder where the repository is copied, run
  ```sh
  ./train_LightOCT_retinal.sh -w path/to/where/the/repository/is/located -d /path/to/the/per_class/retinal/dataset -g 0
  ```
This will:
1. Evoke the **configure_training.py** script that takes, among others, the path to the folder where the different scripts are located (whereafter working directory, parameter **-w**) and the to the folder where the dataset is located (parameter **-d**). The last parapeter (**-g**) specifies on which GPU the code will be run. Note that the dataset path has to match the type of splitting strategy that is set. In the case of dataset_split_strategy=*original*, the dataset path should point to the dataset as it was downloaded. In the case dataset_split_strategy=*per-image* or *per-volume*, the path should point to the dataset organized as per-class (the one obtained using the **refine_dataset.py** script).\
The **configure_training.py** splits the dataset using the specified strategy, sets the model parameters based on the dataset configuration and saves the configuration information in a **config.json** file. The **config.json** file is saved in the working directory under *trained_models/Model_name*
2. Run the **run_training.py** script that, using the **config.json** file, trains the LightOCT model.
6. Test both the best and the last models trained using early stopping (best model = model with highest accuracy, last model = the actual last model before early stopping stopped the training). The test is run by invoking the **test_model.py** script.

Models and training performances are saved independently for every fold in a folder dedicated for the model just trained (*trained_models/Model_name*). Test results are also saved in the dedicated folder as ROC and PP curves, as well as the raw model predictions for the test dataset.

Use 
  ```sh
  ./train_LightOCT_AIIMS.sh -w path/to/where/the/repository/is/located -d /path/to/the/per_class/AIIMS/dataset -g 0
  ```
to run the same routine as before on the AIIMS dataset.

### WINDOWS based operating systems
For users using windows based systems, the **configure_training.py**, **run_training.py** and **test_model.py** scripts can be run independently from comand line. For an understanding of the required and default parameters, run *python3 script_name.py --help*.

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Iulian Emil Tampu - iulian.emil.tampu@liu.se

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
[1] Butola *et al.,* Deep learning architecture “LightOCT” for diagnostic decision support using optical coherence tomography images of biological samples. Biomedical Optics Express. 2020 Sep 1;11(9):5017-31.\
[2] Butola *et al.,* Volumetric analysis of breast cancer tissues using machine learning and swept-source optical coherence tomography. Applied optics. 2019 Feb 10;58(5):A135-41.\
[3] Kermany *et al.,* Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell. 2018 Feb 22;172(5):1122-31.\
[4] Kermany *et al.,* Large dataset of labeled optical coherence tomography (oct) and chest x-ray images. Mendeley Data. 2018;3:10-7632.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->>
<!-- [contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png -->

<!-- ROADMAP 
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

