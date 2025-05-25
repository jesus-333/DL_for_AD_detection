"""
Compute the distribution of the values inside a single dataset.
The data are not divided by class. If you wish to visualize the distribution for each class use distribution_data_V2.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import toml
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import support_dataset_kaggle

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_dataset = 'kaggle'
load_data_in_memory = True
grey_scale_image = True

plot_config = dict(
    figsize = (15, 10),
    fontsize = 15,
    n_bins = 200,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data

path_config_train_and_dataset = './scripts/training/config/demnet_training_and_dataset.toml'
path_config_model             = './scripts/training/config/demnet_model.toml'

path_files_Moderate_Demented    = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/ModerateDemented'
path_files_Mild_Demented        = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/MildDemented'
path_files_Very_Mild_Demented   = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/VeryMildDemented'
path_files_Non_Demented         = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/NonDemented'

train_and_dataset_config = toml.load(path_config_train_and_dataset)
training_config          = train_and_dataset_config['training_config']
dataset_config           = train_and_dataset_config['dataset_config']
model_config             = toml.load(path_config_model)

file_path_list, label_list_int, label_list_str = support_dataset_kaggle.get_data(path_files_Moderate_Demented, path_files_Mild_Demented, path_files_Very_Mild_Demented, path_files_Non_Demented,
                                                                                 dataset_config['merge_AD_class'], print_var = True)

MRI_Dataset = support_dataset_kaggle.get_dataset_with_preprocess_function_from_data(file_path_list, label_list_int, model_config, dataset_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

