"""
With this script you can train the DEMNET model to classify MRI and fMRI data for alzheimer detection.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692


@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# This specific import was added to allow the execution of the script with the python command from the root folder of the repository
# If the code is execute from another folder you have to modifu the path appended to include the src folder
# The path is meant from where you run the python command
import sys
sys.path.append('./')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas
import toml
import torch
import torchvision

from src.dataset import dataset, support_dataset
from src.model import demnet
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings
print("hello world")

path_config_train_and_dataset = './scripts/training/config/demnet_training_and_dataset.toml'
path_config_model             = './scripts/training/config/demnet_model.toml'

dataset_name = 'ADNI_axial_PD_z_44_slice_4'
path_to_data = f'./data/{dataset_name}_png_V4_3/'
z_matrix = int(dataset_name.split('_')[4])
# slice    = int(dataset_name.split('_')[-1])

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load train and dataset config
train_and_dataset_config = toml.load(path_config_train_and_dataset)
train_config = train_and_dataset_config['train_config']
dataset_config = train_and_dataset_config['dataset_config']

# Load model config
model_config = toml.load(path_config_model)
model_config['input_channels'] = z_matrix

# Create single dictionary with all the config
all_config = dict(
    train_config = train_config,
    dataset_config = dataset_config,
    model_config = model_config
)

if 'path_to_data' in dataset_config : path_to_data = dataset_config['path_to_data']

# train_config['epoch_to_save_model'] = train_config['epochs'] + 2

# Note that toml file din't have (yet) the null type
if train_config['seed'] == -1 : train_config['seed'] = np.random.randint(0, 1e9)

# Wand Setting
train_config['wandb_training'] = True
train_config['project_name'] = "demnet_training_ADNI"
train_config['name_training_run'] = None
train_config['model_artifact_name'] = f"demnet_ADNI_z_{z_matrix}"

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files divided per folder

# Get mean and std for normalization
mean = torch.load(f'{path_to_data}dataset_mean.pt')
std = torch.load(f'{path_to_data}dataset_std.pt')
preprocess_functions  = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = mean, std = std)])

# Get data
data = torch.load(f'{path_to_data}dataset_tensor___176_resize___pixel_rescaling.pt')
data = data.type(torch.float) / 4095

# Get labels
dataset_info = pandas.read_csv(f'{path_to_data}info_dataframe.csv')
labels = dataset_info['labels_int'].to_numpy()
labels_str = dataset_info['labels_str'].to_numpy()

if dataset_config['merge_AD_class'] == 1 :
    label_to_int = dict(
        CN    = 0,
        AD    = 1,
        MCI   = 1,
        EMCI  = 1,
        LMCI  = 1,
        SMC   = 1,
    )
    for i in range(len(labels)) : labels[i] = label_to_int[labels_str[i]]
elif dataset_config['merge_AD_class'] == 2 :
    label_to_int = dict(
        CN    = 0,
        AD    = 1,
        MCI   = 2,
        EMCI  = 2,
        LMCI  = 2,
        SMC   = 3,
    )
    for i in range(len(labels)) : labels[i] = label_to_int[labels_str[i]]

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels, percentage_split_list, train_config['seed'])
idx_train, idx_validation, idx_test = idx_list

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels, percentage_split_list, train_config['seed'])
idx_train, idx_validation, idx_test = idx_list

# Save indices in the config
train_config['idx_train']      = idx_train
train_config['idx_test']       = idx_test
train_config['idx_validation'] = idx_validation

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
train_config['device'] = device

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load model
model_config['num_classes'] = len(np.unique(labels))
model = demnet.demnet(model_config)

# Split data in train/validation/test
MRI_train_dataset      = dataset.MRI_dataset(data[idx_train],      labels[idx_train],      preprocess_functions = preprocess_functions)
MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation], labels[idx_validation], preprocess_functions = preprocess_functions)
MRI_test_dataset       = dataset.MRI_dataset(data[idx_test],       labels[idx_test],       preprocess_functions = preprocess_functions)
print("\nDataset split in train/validation/test")
print(f"\tTrain samples      = {len(MRI_train_dataset)}")
print(f"\tValidation samples = {len(MRI_validation_dataset)}")
# print(f"\tTest samples       = {len(MRI_test_dataset)}")

# Delete original data tensor to free memory
del data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model

model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset)
