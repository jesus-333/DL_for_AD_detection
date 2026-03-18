"""
Script used to train demnet. It is launched from the sh scripts in the folders `./scripts_sh/train_demnet_FL_V2/` and `./scripts_sh/train_demnet_FL_V2_IND/`

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments and check

import argparse

parser = argparse.ArgumentParser(description = 'Train dement model in a centralized way.')
parser.add_argument('--path_src'                         , type = str  , default = None, help = 'Path of the src folder (i.e. all the code related to model, dataset and training.)')
parser.add_argument('--path_dataset_config'              , type = str  , default = None, help = 'Path to the dataset toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_model_config'                , type = str  , default = None, help = 'Path to the model toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_training_config'             , type = str  , default = None, help = 'Path to the training toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_to_idx_files'                , type = str  , default = None, help = 'Path to the folder with the idx files. If not provided, an error will be thrown. The folder must contain two files named train_idx_all.npy and val_idx.npy. The two files are created by the script create_idx_files_for_federated_simulations_2.py')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# This specific import was added to allow the execution of the script with the "python" command from any folder you like.
# If the argument path_src is not provided, the script assume you will run it from the root folder of the repository.
import sys
if args.path_src is not None : sys.path.append(args.path_src)
else : sys.path.append('./')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pandas as pd
import pprint
import toml
import torch

from src.dataset import support_dataset_ADNI
from src.model import demnet
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Default settings. Used only if the corresponding argument is not provided when launching the script.

# Path to config files.
defualt_path_config_dataset  = './config/demnet_wandb/dataset.toml'
defualt_path_config_model    = './config/demnet_wandb/model.toml'
defualt_path_config_training = './config/demnet_wandb/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load config

path_config_dataset  = args.path_dataset_config if args.path_dataset_config is not None else defualt_path_config_dataset
path_config_model    = args.path_model_config if args.path_model_config is not None else defualt_path_config_model
path_config_training = args.path_training_config if args.path_training_config is not None else defualt_path_config_training

# Load configs
dataset_config  = toml.load(path_config_dataset)
model_config    = toml.load(path_config_model)
training_config = toml.load(path_config_training)

# Create single dictionary with all the config
all_config = dict(
    training_config = training_config,
    dataset_config = dataset_config,
    model_config = model_config
)

pprint.pprint(all_config)

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
# Load data

dataset_info = pd.read_csv(f'{dataset_config['path_data']}dataset_info.csv')
labels_int, labels_str = dataset_info['labels_int'].to_numpy(), dataset_info['labels_str'].to_numpy()
labels_int = support_dataset_ADNI.merge_AD_class_function(labels_int, labels_str, dataset_config['merge_AD_class'])
num_classes = len(np.unique(labels_int))

# Get the indices for training and validation (the two files must be created before launching the script with the script create_idx_files_for_federated_simulations_2.py)
idx_train = np.load(f"{args.path_to_idx_files}train_idx_all.npy")
idx_val   = np.load(f"{args.path_to_idx_files}val_idx.npy")

# Get train and validation dataset
MRI_train_dataset, _, _      = support_dataset_ADNI.get_dataset_V2(dataset_config, idx_to_use = idx_train)
MRI_validation_dataset, _, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, idx_to_use = idx_val)

if training_config['print_var'] :
    print("#######################################")
    print("Demnet Centralized Training")
    print("dataset_config['merge_AD_class'] ", dataset_config['merge_AD_class'])
    print(f"N. training samples    : {len(MRI_train_dataset)}")
    print(f"N. validations samples : {len(MRI_validation_dataset)}")
    print("#######################################")

# (OPTIONAL) Move dataset to device
if dataset_config['load_data_in_memory'] :
    MRI_train_dataset.move_data_and_labels_to_device(training_config['device'])
    MRI_validation_dataset.move_data_and_labels_to_device(training_config['device'])

print("Data Loaded")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load model
model = demnet.demnet(model_config)

print("Model Loaded")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model
model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset)
