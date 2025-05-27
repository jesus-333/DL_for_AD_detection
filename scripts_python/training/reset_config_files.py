"""
Reset all the parameters inside toml config files for training and dataset to default values.
The model config will not be resetted since each model has its own config file. And therefore requires itw own reset script.
The only input parameters of this script are the path to the toml config files
If not provided a training.toml file will be created in the directory './config/'

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

import argparse
import os
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Parser
parser = argparse.ArgumentParser(description = 'Reset training and dataset configuration files to default values.')
parser.add_argument('--path_training_config', type = str, default = './config/training.toml', help = 'Path to the toml file with the training config. Default is ./config/training.toml')
parser.add_argument('--path_dataset_config' , type = str, default = './config/dataset.toml' , help = 'Path to the toml file with the dataset config. Default is ./config/dataset.toml')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load the config files
# For each config file it tries to load it. If the file does not exist, it creates a new one with default values.

# Training config
path_training_config   = args.path_training_config
folder_training_config = os.path.dirname(path_training_config)
try :
    training_config = toml.load(path_training_config)
except FileNotFoundError :
    print(f"Training config file not found at {path_training_config}. Creating a new one with default values at {folder_training_config}.")
    os.makedirs(folder_training_config, exist_ok = True)
    training_config = {}

# Dataset config
path_dataset_config   = args.path_dataset_config
folder_dataset_config = os.path.dirname(path_dataset_config)
try :
    dataset_config = toml.load(path_dataset_config)
except FileNotFoundError :
    print(f"Dataset config file not found at {path_dataset_config}. Creating a new one with default values at {folder_dataset_config}.")
    os.makedirs(folder_dataset_config, exist_ok = True)
    dataset_config = {}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reset training config

# Reset values
training_config['batch_size']                      = 64
training_config['lr']                              = 0.001
training_config['epochs']                          = 40
training_config['use_scheduler']                   = False
training_config['device']                          = 'cpu'
training_config['epoch_to_save_model']             = -1  # Save only the final model
training_config['path_to_save_model']              = 'model_weights'
training_config['measure_metrics_during_training'] = True
training_config['print_var']                       = True
training_config['seed']                            = -1  # Random seed
training_config['wandb_training']                  = False
training_config['project_name']                    = None
training_config['model_artifact_name']             = None
training_config['name_training_run']               = None
training_config['notes']                           = ''
training_config['log_freq']                        = 1
training_config['debug']                           = False

# Save the training config to the file
with open(path_training_config, 'w') as f: toml.dump(training_config, f)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reset dataset config

# Reset values
dataset_config['use_normalization']     = True
dataset_config['n_samples']             = 19200
dataset_config['merge_AD_class']        = 0
dataset_config['load_data_in_memory']   = False
dataset_config['load_data_type']        = 0  # 0: load data from disk, 1: load data in memory
dataset_config['percentage_train']      = 0.7
dataset_config['percentage_validation'] = 0.15
dataset_config['percentage_test']       = 0.15

# Save the dataset config to the file
with open(path_dataset_config, 'w') as f: toml.dump(dataset_config, f)
