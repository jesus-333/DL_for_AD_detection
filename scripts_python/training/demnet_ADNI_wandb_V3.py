"""
With this script you can train the DEMNET model to classify MRI and fMRI data for alzheimer detection.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692


@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (OPTIONAL) CLI argument
import argparse

# Create parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--path_src'              , type = str, default = None, help = 'Path of the src folder (i.e. all the code related to model, dataset and training.)')
parser.add_argument('--path_training_config'  , type = str, default = None, help = 'Path to the toml file with the training config. If not provided, the script will use the default value of ./config/training.toml')
parser.add_argument('--path_dataset_config'   , type = str, default = None, help = 'Path to the toml file with the dataset config. If not provided, the script will use the default value of ./config/dataset.toml')
parser.add_argument('--path_model_config'     , type = str, default = None, help = 'Path to the toml file with the model config. If not provided, the script will use the default value of ./config/model.toml')
parser.add_argument('--path_data'             , type = str, default = None, help = 'Path to the folder with the data. If not provided, it will use the value defined in this script.')
parser.add_argument('--name_tensor_file'      , type = str, default = None, help = 'Name of the tensor file with the dataset. If not provided, it will use the default value defined in this script.')
args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# This specific import was added to allow the execution of the script with the "python" command from any folder you like.
# If the argument path_src is not provided, the script assume you will run it from the root folder of the repository.
import sys
if args.path_src is not None : sys.path.append(args.path_src)
else : sys.path.append('./')

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

path_config_dataset  = args.path_dataset_config if args.path_dataset_config is not None else './config/demnet_wandb/dataset.toml'
path_config_model    = args.path_model_config if args.path_model_config is not None else './config/demnet_wandb/model.toml'
path_config_training = args.path_training_config if args.path_training_config is not None else './config/demnet_wandb/training.toml'

dataset_name = 'ADNI_axial_middle_slice'
dataset_tensor_file_name = 'dataset_tensor___176_resize.pt' if args.name_tensor_file is None else args.name_tensor_file
path_to_data = f'./data/{dataset_name}/' if args.path_data is None else args.path_data

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load configs
training_config = toml.load(path_config_training)
dataset_config  = toml.load(path_config_dataset)
model_config    = toml.load(path_config_model)

# Create single dictionary with all the config
all_config = dict(
    training_config = training_config,
    dataset_config = dataset_config,
    model_config = model_config
)
import pprint
pprint.pprint(all_config)

if 'path_to_data' in dataset_config : path_to_data = dataset_config['path_to_data']

# training_config['epoch_to_save_model'] = training_config['epochs'] + 2

# Note that toml file din't have (yet) the null type
if training_config['seed'] == -1 : training_config['seed'] = np.random.randint(0, 1e9)

# Wand Setting
training_config['wandb_training'] = True
# training_config['project_name'] = "demnet_training_ADNI"
# training_config['name_training_run'] = None

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files divided per folder

# Get mean and std for normalization
if dataset_config['use_normalization'] :
    # Load procumputed dataset mean and std
    mean_dataset = torch.load(f'{path_to_data}dataset_mean.pt')
    std_dataset  = torch.load(f'{path_to_data}dataset_std.pt')
    # Save them in the config (In this way I save them also on wandb)
    dataset_config['mean_dataset'] = mean_dataset
    dataset_config['std_dataset']  = std_dataset
    # Create normalization function
    preprocess_functions  = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = mean_dataset, std = std_dataset)])
else :
    preprocess_functions = None

# Get data
data = torch.load(f'{path_to_data}{dataset_tensor_file_name}', mmap = True)

# Get the number of channels
model_config['input_channels'] = data.shape[1]

# Get labels
dataset_info = pd.read_csv(f'{path_to_data}dataset_info.csv')
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
idx_list = support_dataset.get_idx_to_split_data_V3(labels, percentage_split_list, training_config['seed'])
idx_train, idx_validation, idx_test = idx_list

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels, percentage_split_list, training_config['seed'])
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
model_config['num_classes'] = len(np.unique(labels))
model = demnet.demnet(model_config)

# Split data in train/validation/test
if dataset_config['apply_rescale'] :
    # Not that I apply the rescale here to avoid the increase in memory before.
    # This is valid only if the test set is not created.
    # If the data are saved in int16 to save memory, the rescale in the 0-1 range by deafult cast them to float32 and therefore double the memory usage.
    # If you don't need the test set, as in this case, you can cast to float32 only the train and validation set, reducing the memory consumption.

    MRI_train_dataset      = dataset.MRI_dataset(data[idx_train] / dataset_config['rescale_factor']     , labels[idx_train]     , preprocess_functions = preprocess_functions)
    MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation] / dataset_config['rescale_factor'], labels[idx_validation], preprocess_functions = preprocess_functions)
    # MRI_test_dataset       = dataset.MRI_dataset(data[idx_test] / dataset_config['rescale_factor']      , labels[idx_test]      , preprocess_functions = preprocess_functions)
else :
    MRI_train_dataset      = dataset.MRI_dataset(data[idx_train]     , labels[idx_train]     , preprocess_functions = preprocess_functions)
    MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation], labels[idx_validation], preprocess_functions = preprocess_functions)
    # MRI_test_dataset       = dataset.MRI_dataset(data[idx_test]      , labels[idx_test]      , preprocess_functions = preprocess_functions)
print("\nDataset split in train/validation/test")
print(f"\tTrain samples      = {len(MRI_train_dataset)}")
print(f"\tValidation samples = {len(MRI_validation_dataset)}")
# print(f"\tTest samples       = {len(MRI_test_dataset)}")

# Delete original data tensor to free memory
del data

# (OPTIONAL) Move dataset to device
if dataset_config['load_data_in_memory'] :
    MRI_train_dataset.move_data_and_labels_to_device(device)
    MRI_validation_dataset.move_data_and_labels_to_device(device)
    # MRI_test_dataset.move_data_and_labels_to_device(device)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model
# model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset)
