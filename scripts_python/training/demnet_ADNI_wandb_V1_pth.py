"""
With this script you can train the DEMNET model to classify MRI and fMRI data for alzheimer detection.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692

This version is used to training the model on THE converted ADNI Dataset. The data are obtained through the script convert/all_subjects_ADNI_ONLY_2D_MRI_V5_unpack.py and ADNI_extract_middle_slice.py
The data are store in pth tensor.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd
import toml
import torch
import torchvision

from src.dataset import dataset, support_dataset
from src.model import demnet
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_config_train_and_dataset = './scripts/training/config/demnet_training_and_dataset.toml'
path_config_model             = './scripts/training/config/demnet_model.toml'

path_to_data_folder = './data/ADNI_axial_middle_slice/'
path_to_data_tensor = f'{path_to_data_folder}dataset_tensor___176_resize.pt'

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load train and dataset config
train_and_dataset_config = toml.load(path_config_train_and_dataset)
training_config = train_and_dataset_config['training_config']
dataset_config = train_and_dataset_config['dataset_config']

# Load model config
model_config = toml.load(path_config_model)
model_config['input_channels'] = 1 if dataset_config['grey_scale_image'] else 3

# Create single dictionary with all the config
all_config = dict(
    training_config = training_config,
    dataset_config = dataset_config,
    model_config = model_config
)

if training_config['seed'] == -1 : training_config['seed'] = np.random.randint(0, 1e9)
dataset_config['grey_scale_image'] = True

if dataset_config['use_normalization'] :
    # Load precomputed mean and std
    mean_to_use = torch.load(f'{path_to_data_folder}dataset_mean.pt')
    std_to_use  = torch.load(f'{path_to_data_folder}dataset_std.pt')

    # Create preprocess function
    dataset_mean = torch.tensor([mean_to_use, mean_to_use, mean_to_use]) if not dataset_config['grey_scale_image'] else torch.tensor([mean_to_use])
    dataset_std  = torch.tensor([std_to_use, std_to_use, std_to_use]) if not dataset_config['grey_scale_image'] else torch.tensor([std_to_use])
    preprocess_functions  = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = dataset_mean, std = dataset_std)])
else :
    preprocess_functions = None

# Save in the settings dataset_mean and dataset_std
if dataset_config['use_normalization'] :
    dataset_config['dataset_mean'] = dataset_mean
    dataset_config['dataset_std'] = dataset_std

# Wand Setting
training_config['wandb_training'] = True
training_config['project_name'] = "demnet_training_ADNI_2D"
training_config['name_training_run'] = None
training_config['model_artifact_name'] = "demnet_training_ADNI"

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data path
data = torch.load(path_to_data_tensor)
dataset_info = pd.read_csv(f'{path_to_data_folder}data_info.csv')

label_list_int = dataset_info['labels_int'].to_numpy()
label_list_str = dataset_info['labels_str'].to_numpy()

if dataset_config['merge_AD_class'] == 1 :
    label_to_int = dict(
        CN    = 0,
        AD    = 1,
        MCI   = 1,
        EMCI  = 1,
        LMCI  = 1,
        SMC   = 1,
    )
    for i in range(len(label_list_int)) : label_list_int[i] = label_to_int[label_list_str[i]]
elif dataset_config['merge_AD_class'] == 2 :
    label_to_int = dict(
        CN    = 0,
        AD    = 1,
        MCI   = 2,
        EMCI  = 2,
        LMCI  = 2,
        SMC   = 3,
    )
    for i in range(len(label_list_int)) : label_list_int[i] = label_to_int[label_list_str[i]]
else :
    label_to_int = dict(
        CN    = 0,
        AD    = 1,
        MCI   = 2,
        EMCI  = 3,
        LMCI  = 4,
        SMC   = 5,
    )


# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(label_list_int, percentage_split_list, training_config['seed'])
idx_train, idx_validation, idx_test = idx_list

# Save indices in the config
training_config['idx_train']      = idx_train
training_config['idx_test']       = idx_test
training_config['idx_validation'] = idx_validation

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Select training device

if torch.cuda.is_available() :
    device = torch.device("cuda")
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nmps backend (apple metal) in use")
else:
    device = torch.device("cpu")
    print("\nNo backend in use. Device set to cpu")
training_config['device'] = device

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load model
model_config['num_classes'] = len(set(label_list_int))
model = demnet.demnet(model_config)

# Create datasets
load_data_in_memory = dataset_config['load_data_in_memory']
MRI_train_dataset      = dataset.MRI_dataset(data[idx_train],      label_list_int[idx_train],      preprocess_functions = preprocess_functions)
MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation], label_list_int[idx_validation], preprocess_functions = preprocess_functions)
MRI_test_dataset       = dataset.MRI_dataset(data[idx_test],       label_list_int[idx_test],       preprocess_functions = preprocess_functions)
print("\nDatasets CREATED")
print(f"\tTrain samples      = {len(MRI_train_dataset)}")
print(f"\tValidation samples = {len(MRI_validation_dataset)}")
# print(f"\tTest samples       = {len(MRI_test_dataset)}")

# Delete original data tensor to free memory
del data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model

model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset) 
