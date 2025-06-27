"""
With this script you can train/finetune the VGG model to classify MRI data for alzheimer detection.

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
print("Parser - OK")

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

from src.dataset import dataset, support_dataset, support_dataset_ADNI
from src.model import download_published_model, vgg_nets
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_config_dataset  = args.path_dataset_config if args.path_dataset_config is not None else './config/vgg_wandb/dataset.toml'
path_config_model    = args.path_model_config if args.path_model_config is not None else './config/vgg_wandb/model.toml'
path_config_training = args.path_training_config if args.path_training_config is not None else './config/vgg_wandb/training.toml'

dataset_name = 'ADNI_axial_3D_z_48_size_176_int'
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
# training_config['project_name'] = "vgg_training_ADNI"
# training_config['name_training_run'] = None

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

print("Config loaded")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files divided per folder

# Get mean and std for normalization
if dataset_config['use_normalization'] :
    if training_config['use_vgg_normalization_values'] :
        preprocess_functions = download_published_model.get_preprocess_functions('vgg')

        # Save them in the config (In this way I save them also on wandb)
        dataset_config['mean_dataset'] = torch.tensor([0.485, 0.456, 0.406])
        dataset_config['std_dataset']  = torch.tensor([0.229, 0.224, 0.225])
    else :
        mean_dataset = torch.load(f'{path_to_data}dataset_mean.pt')
        std_dataset  = torch.load(f'{path_to_data}dataset_std.pt')
        preprocess_functions = download_published_model.get_preprocess_functions('vgg', mean = mean_dataset, std = std_dataset)

        # Save them in the config (In this way I save them also on wandb)
        dataset_config['mean_dataset'] = mean_dataset
        dataset_config['std_dataset']  = std_dataset
else :
    preprocess_functions = None

# Get data
data = torch.load(f'{path_to_data}{dataset_tensor_file_name}', mmap = True)
print("Data loaded")

# Get the number of channels
model_config['input_channels'] = data.shape[1]

# Check the number of channels. If it is 1, modify the data to have 3 channels.
# Works only if the data are grayscale (i.e. 1 channel). The single channel is repeated 3 times to create a 3 channels image.
if model_config['input_channels'] == 3 and data.shape[1] == 1 :
    print("VGG created with 3 input cahnnels, but the data have only 1 channel. Therefore, I will repeat the single channel 3 times.")
    data = data.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
elif model_config['input_channels'] == 3 and data.shape[1] not in [1, 3] :
    raise ValueError(f"If model_config['input_channels'] is 3, the data must have 1 or 3 channels. Current number of channels in the data: {data.shape[1]}")

# Get labels
dataset_info = pd.read_csv(f'{path_to_data}dataset_info.csv')
labels_int = dataset_info['labels_int'].to_numpy()
labels_str = dataset_info['labels_str'].to_numpy()
print("Labels loaded")

# (OPTIONAL) Filter AD data
if dataset_config['filter_AD_data'] :
    index_to_keep = support_dataset_ADNI.filter_AD_data(labels_str, dataset_config['class_to_keep'], return_boolean_index = True)
    data       = data[index_to_keep]
    labels_in  = labels_int[index_to_keep]
    labels_str = labels_str[index_to_keep]

# Check input channels
if dataset_config['use_rgb_input'] and data.shape[1] == 1 :
    # Convert to 3 channels if greyscale images are used and use_rgb_input is set to True
    print("Data have only 1 channel, but use_rgb_input is set to True. Therefore, I will repeat the single channel 3 times.")

    # Repeat the single channel 3 times
    data = data.repeat(1, 3, 1, 1)

    # Update the input channels in the model config
    model_config['input_channels'] = 3
elif dataset_config['use_rgb_input'] and data.shape[1] not in [1, 3] :
    raise ValueError(f"If dataset_config['use_rgb_input'] is True, the data must have 1 or 3 channels. Current number of channels in the data: {data.shape[1]}")

# (OPTIONAL) Merge AD classes
labels_int = support_dataset_ADNI.merge_AD_class_function(labels_int, labels_str, dataset_config['merge_AD_class'])

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels_int, percentage_split_list, training_config['seed'])
idx_train, idx_validation, idx_test = idx_list

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels_int, percentage_split_list, training_config['seed'])
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
model_config['num_classes'] = len(np.unique(labels_int))
model_config['use_pretrained_vgg'] = training_config['use_pretrained_vgg']
model = vgg_nets.get_vgg(model_config, return_preprocess_functions = False)
model.set_training_mode(training_config['vgg_training_mode'])
print("Model created")

# Split data in train/validation/test
if dataset_config['apply_rescale'] :
    # Not that I apply the rescale here to avoid the increase in memory before.
    # This is valid only if the test set is not created.
    # If the data are saved in int16 to save memory, the rescale in the 0-1 range by deafult cast them to float32 and therefore double the memory usage.
    # If you don't need the test set, as in this case, you can cast to float32 only the train and validation set, reducing the memory consumption.

    MRI_train_dataset      = dataset.MRI_dataset(data[idx_train] / dataset_config['rescale_factor']     , labels_int[idx_train]     , preprocess_functions = preprocess_functions)
    MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation] / dataset_config['rescale_factor'], labels_int[idx_validation], preprocess_functions = preprocess_functions)
    # MRI_test_dataset       = dataset.MRI_dataset(data[idx_test] / dataset_config['rescale_factor']      , labels[idx_test]      , preprocess_functions = preprocess_functions)
else :
    MRI_train_dataset      = dataset.MRI_dataset(data[idx_train]     , labels_int[idx_train]     , preprocess_functions = preprocess_functions)
    MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation], labels_int[idx_validation], preprocess_functions = preprocess_functions)
    # MRI_test_dataset       = dataset.MRI_dataset(data[idx_test]      , labels_int[idx_test]      , preprocess_functions = preprocess_functions)
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
    print(f"Dataset moved to {device}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model
model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset)
