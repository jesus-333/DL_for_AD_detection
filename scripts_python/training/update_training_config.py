"""
Update the training config toml file

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import numpy as np
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the training configuration file with new parameters.')

# Add arguments


# *******************************
# General arguments (i.e. arguments valid for all models and training type)
# Various arguments
parser.add_argument('--path_training_config'           , type = str  , default = './config/training.toml'    , help = 'Path to the toml file with the training config. Default is ./config/training.toml')
parser.add_argument('--path_lr_scheduler_config'       , type = str  , default = './config/lr_scheduler.toml', help = 'Path to the toml file with the learning rate scheduler config. Default is ./config/lr_scheduler.toml')
parser.add_argument('--batch_size'                     , type = int  , default = -1             , help = 'Batch size for training. If a negative value (or no value) is provided, the value already present will not be changed. If a positive value is provided, it will be used as the new batch size. Default is -1 (do not change).')
parser.add_argument('--lr'                             , type = float, default = -1             , help = 'Learning rate for the optimizer. If a negative value (or no value) is provided, the value already present will not be changed. If a positive value is provided, it will be used as the new learning rate. Default is -1 (do not change).')
parser.add_argument('--epochs'                         , type = int  , default = -1             , help = 'Number of epochs for training. If a negative value (or no value) is provided, the value already present will not be changed. If a positive value is provided, it will be used as the new number of epochs. Default is -1 (do not change).')
parser.add_argument('--device'                         , type = str  , default = 'cpu'          , help = 'Device to use for training. Default is "cpu".')
parser.add_argument('--epoch_to_save_model'            , type = int  , default = -1             , help = 'Save model every n epochs. If a negative value (or no value) is provided, it will be set to epochs + 1, i.e. only the model at the end of training will be saved. If a positive value is provided, it will be used as the new value. Default is -1.')
parser.add_argument('--path_to_save_model'             , type = str  , default = 'model_weights', help = 'Path to save the model weights. If the folder does not exist, it will be created. Default is "model_weights".')
parser.add_argument('--seed'                           , type = int  , default = -1             , help = 'Seed for reproducibility. It is used to split the dataset. If a negative value (or no value) is provided, the seed will be set to a random value. Default is -1.')
# Boolean arguments
parser.add_argument('--use_scheduler'                  , default = True  , action = "store_true", help = 'If True, use a learning rate scheduler. Default is True.')
parser.add_argument('--measure_metrics_during_training', default = True  , action = "store_true", help = 'Measure metrics during training. If True various secondary metrics (e.g. accuracy, f1 score, etc.) will be computed during training. Default is True.')
parser.add_argument('--print_var'                      , default = True  , action = "store_true", help = 'If True print information during training. Default is True.')
parser.add_argument('--wandb_training'                 , default = False , action = "store_true", help = 'If True, use Weights & Biases (wandb) for tracking the training. Default is False.')
parser.add_argument('--debug'                          , default = False , action = "store_true", help = 'Used only as a flag to quickly find runs in wandb. Used to test the code. Default is False.')
# Boolen negate
parser.add_argument('--no-use_scheduler'                  , dest ='use_scheduler'                  , action = 'store_false')
parser.add_argument('--no-debug'                          , dest ='feature'                        , action = 'store_false')
parser.add_argument('--no-measure_metrics_during_training', dest ='measure_metrics_during_training', action = 'store_false')
parser.add_argument('--no-print_var'                      , dest ='print_var'                      , action = 'store_false')
parser.add_argument('--no-wandb_training'                 , dest ='wandb_training'                 , action = 'store_false')
# Wandb settings
parser.add_argument('--project_name'       , type = str, default = None, help = 'Name of the wandb project. Default is None.')
parser.add_argument('--model_artifact_name', type = str, default = None, help = 'Name of the wandb model artifact. Default is None.')
parser.add_argument('--name_training_run'  , type = str, default = None, help = 'Name of the training run in wandb. Default is None.')
parser.add_argument('--notes'              , type = str, default = None, help = 'Notes for the training run in wandb. Default is None.')
parser.add_argument('--log_freq'           , type = int, default = 1   , help = 'Frequency of wandb logging during training. Default is 1 (every epoch).')

# *******************************
# Arguments for Federated Learning only
parser.add_argument('--fl_training'                               , default = False, action = "store_true", help = "If True, the training is done in Federated Learning mode. Default is False.")
parser.add_argument('--use_weights_with_lower_validation_error'   , default = False, action = "store_true" , help = "This value is used only during FL training. If True, each client will send to the central server the weights that achieve the lowest validation error, if False the weights at the end of training will be sent. Default is False.")
parser.add_argument('--no-use_weights_with_lower_validation_error', dest ='use_weights_with_lower_validation_error', action = 'store_false')

# *******************************
# VGG training arguments
parser.add_argument('--vgg_training'                   , default = False, action = "store_true", help = "If True, the training is done using a VGG network. Default is False.")
parser.add_argument('--use_vgg_normalization_values'   , default = True , action = "store_true", help = "If True, when vgg is trained, the data are normalized using the values used in the original VGG paper. Default is None")
parser.add_argument('--no-use_vgg_normalization_values', dest = 'use_vgg_normalization_values' , action = 'store_false')
parser.add_argument('--vgg_training_mode'              , type = int, default = 0, help = "Training mode for the VGG network. Possible values are 0, 1, 2 or 3. See set_training_model method in the VGG class for more details on the training modes. Note that this argument is used only if the VGG network is trained, i.e. if the model is a VGG network.")

# *******************************

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Update the training config

# Load the training config from the provided path
training_config = toml.load(args.path_training_config)

# Batch size
if args.batch_size is not None and args.batch_size > 0 :
    training_config['batch_size'] = args.batch_size
else :
    print(f"Invalid batch size provided: {args.batch_size}. Using value from config: {training_config['batch_size']}.")

# Learning rate
if args.lr is not None and args.lr > 0 :
    training_config['lr'] = args.lr
else :
    print(f"Invalid learning rate provided: {args.lr}. Using value from config: {training_config['lr']}.")

# Number of epochs
if args.epochs is not None and args.epochs > 0 :
    training_config['epochs'] = args.epochs
else :
    print(f"Invalid number of epochs provided: {args.epochs}. Using value from config: {training_config['epochs']}.")

# Device
if args.device is not None and args.device in ['cpu', 'cuda', 'mps'] :
    training_config['device'] = args.device
else :
    print(f"Invalid device provided: {args.device}. Accepted values are 'cpu', 'cuda', or 'mps'. Using value from config: {training_config['device']}.")

# Epoch to save model
if args.epoch_to_save_model is not None and args.epoch_to_save_model > 0 :
    training_config['epoch_to_save_model'] = args.epoch_to_save_model
    print(f"Model will be saved every {training_config['epoch_to_save_model']} epochs.")
else :
    training_config['epoch_to_save_model'] = training_config['epochs'] + 1
    print("Model will ba saved on the end of training.")

# Path to save model
training_config['path_to_save_model'] = args.path_to_save_model
if args.path_to_save_model is None : print("No path provided to save the model. Using default path: 'model_weights'.")

# Measure metrics during training
training_config['measure_metrics_during_training'] = args.measure_metrics_during_training
if args.measure_metrics_during_training is None : print("No value provided for measure_metrics_during_training. Using default value: True.")

# Print variable
training_config['print_var'] = args.print_var
if args.print_var is None : print("No value provided for print_var. Using default value: True.")

# Seed
if args.seed is not None and args.seed >= 0 :
    training_config['seed'] = args.seed
else :
    training_config['seed'] = np.random.randint(0, 1e9)
    print(f"Invalid seed provided: {args.seed}. Using a random seed: {training_config['seed']}.")

# Use scheduler
if args.use_scheduler is not None and isinstance(args.use_scheduler, bool) :
    training_config['use_scheduler'] = args.use_scheduler
else :
    print(f"Invalid use_scheduler value provided: {args.use_scheduler}. Using value from config: {training_config['use_scheduler']}.")

if training_config['use_scheduler'] :
    # Load the learning rate scheduler config from the provided path
    lr_scheduler_config = toml.load(args.path_lr_scheduler_config)
    
    # Set the learning rate scheduler config in the training config
    # Note that this script will simply saved the lr schduler config as it is, without checking/updating the values inside.
    # To update the learning rate scheduler config you could use the script `update_lr_scheduler_config.py`.
    # I keep the scripts separated to avoid a cumbersome script that does everything.
    training_config['lr_scheduler_config'] = lr_scheduler_config
else :
    training_config['lr_scheduler_config'] = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Wandb training
# Check and update config only if wand is training is not False
if args.wandb_training is True :
    training_config['wandb_training'] = True

    # Project name
    if args.project_name is None : print("No project name provided for wandb. Using default value: None.")
    training_config['project_name'] = args.project_name

    # Model artifact name
    if args.model_artifact_name is None : print("No model artifact name provided for wandb. Using default value: None.")
    training_config['model_artifact_name'] = args.model_artifact_name

    # Name of the training run
    if args.name_training_run is None : print("No name provided for the training run in wandb. Using default value: None.")
    training_config['name_training_run'] = args.name_training_run

    # Notes for the training run
    if args.notes is None : print("No notes provided for the training run in wandb. Using default value: None.")
    training_config['notes'] = args.notes

    # Log frequency
    if args.log_freq is not None and args.log_freq > 0 :
        training_config['log_freq'] = args.log_freq
    else :
        training_config['log_freq'] = 1
        print(f"Invalid log frequency provided: {args.log_freq}. Using default value: {training_config['log_freq']}.")

# Debug flag
training_config['debug'] = args.debug

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Update the training config for Federated Learning

if args.fl_training is not None and args.fl_training is True :
    training_config['fl_training'] = True
    training_config['use_weights_with_lower_validation_error'] = args.use_weights_with_lower_validation_error

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Update the training config for VGG training

if args.vgg_training is not None and args.vgg_training is True :
    training_config['vgg_training'] = True

    # Use VGG normalization values (see https://pytorch.org/hub/pytorch_vision_vgg/)
    if args.use_vgg_normalization_values is not None :
        training_config['use_vgg_normalization_values'] = args.use_vgg_normalization_values

    # Training mode for VGG
    if args.vgg_training_mode in [0, 1, 2, 3] :
        training_config['vgg_training_mode'] = args.vgg_training_mode
    else :
        raise ValueError(f"Invalid vgg_training_mode provided: {args.vgg_training_mode}. Possible values are 0, 1, 2 or 3.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Save the updated training config to the same file
with open(args.path_training_config, 'w') as f:
    toml.dump(training_config, f)
