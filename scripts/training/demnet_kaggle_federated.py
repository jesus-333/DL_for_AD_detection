"""
With this script you can train the DEMNET model to classify MRI and fMRI data for alzheimer detection.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692 
For the dataset we used the the Kaggle alzheimer 4 class dataset (https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset/data)

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import toml
import numpy as np
import torch
from torchvision import transforms

from src.dataset import dataset_png, support_dataset, support_dataset_kaggle
from src.model import demnet 
from src.training import train_functions
from src.federated import support_federated

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_config_train_and_dataset = './scripts/training/config/demnet_training_and_dataset.toml'
path_config_model             = './scripts/training/config/demnet_model.toml'

path_files_Moderate_Demented    = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/ModerateDemented'
path_files_Mild_Demented        = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/MildDemented'
path_files_Very_Mild_Demented   = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/VeryMildDemented'
path_files_Non_Demented         = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/NonDemented'

# TODO : move to a toml file
print_var = True
n_client = 4
keep_labels_proportion = True
percentage_split_per_client = [1 / n_client] * n_client

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load train and dataset config
train_and_dataset_config = toml.load(path_config_train_and_dataset)
train_config = train_and_dataset_config['train_config']
dataset_config = train_and_dataset_config['dataset_config']

# Load model config
model_config = toml.load(path_config_model)
model_config['input_channels'] = 1 if dataset_config['grey_scale_image'] else 3

# Create single dictionary with all the config
all_config = dict(
    train_config = train_config,
    dataset_config = dataset_config,
    model_config = model_config
)

# train_config['epoch_to_save_model'] = train_config['epochs'] + 2

# Note that toml file din't have (yet) the null type
if train_config['seed'] == -1 : train_config['seed'] = None

# Get preprocess function (from torchvision transform)
preprocess_function, dataset_mean, dataset_std = support_dataset.get_preprocess_function('kaggle', model_config['input_size'], dataset_config['grey_scale_image'])

# Save in the settings dataset_mean and dataset_std
dataset_config['dataset_mean'] = dataset_mean
dataset_config['dataset_std'] = dataset_std

# Wand Setting
train_config['wandb_training'] = True
train_config['project_name'] = "demnet_training"
train_config['name_training_run'] = None
train_config['model_artifact_name'] = "demnet_training_AD_kaggle"

# Percentage used to split data in train/validation/test
percentage_split_train_validation_test = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Dataset creation

# Get data path
file_path_list, label_list_int, label_list_str = support_dataset_kaggle.get_data(path_files_Moderate_Demented, path_files_Mild_Demented, path_files_Very_Mild_Demented, path_files_Non_Demented, 
                                                                                      dataset_config['merge_AD_class'], print_var)

# Divide data for each client
data_per_client, labels_per_client = support_federated.split_data_for_clients(file_path_list, percentage_split_per_client, train_config['seed'], label_list_int, keep_labels_proportion)
