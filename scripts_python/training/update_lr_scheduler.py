"""
Update the learning rate scheduler config toml file
Note that this script works a little bit differently than uupdate_training_config.py
In the latter, if a config files already exists, it is updated the existing file with the new parameters (if it does not exist of cousre a new toml file is created).
This script instead, always creates a new toml file with the new parameters. If a toml file with the same name already exists, it is overwritten.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import os
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the training configuration file with new parameters.')

# Add arguments
parser.add_argument('--path_lr_scheduler_config', type = str  , default = './config/lr_scheduler.toml', help = 'Path to the toml file with the learning rate scheduler config. Default is ./config/lr_scheduler.toml')
parser.add_argument('--name'                    , type = str  , default = None, help = 'Name of the learning rate scheduler. If None is provided, an error is raised. Default is None.')
parser.add_argument('--gamma'                   , type = float, default = None, help = 'Gamma for ExponentialLR/StepLR/CyclicLR. The parameter is required for ExponentialLR but optional for StepLR and CyclicLR. In the latter case, if None is provided, the default PyTorch value is used. In the case of ExponentialLR, if None is provided, an error is raised. Default is None.')
parser.add_argument('--T_max'                   , type = int  , default = None, help = 'T_max for CosineAnnealingLR. If None is provided, an error is raised. Default is None.')
parser.add_argument('--eta_min'                 , type = float, default = 0.0 , help = 'Eta_min for CosineAnnealingLR/CosineAnnealingWarmRestarts. Default is 0.0.')
parser.add_argument('--T_0'                     , type = int  , default = None, help = 'T_0 for CosineAnnealingWarmRestarts. If None is provided, an error is raised. Default is None.')
parser.add_argument('--T_mult'                  , type = int  , default = 1   , help = 'T_mult for CosineAnnealingWarmRestarts. Default is 1.')
parser.add_argument('--step_size'               , type = int  , default = None, help = 'Step size for StepLR. If None is provided, an error is raised. Default is None.')
parser.add_argument('--base_lr'                 , type = float, default = None, help = 'Base learning rate for CyclicLR. If None is provided, an error is raised. Default is None.')
parser.add_argument('--max_lr'                  , type = float, default = None, help = 'Maximum learning rate for CyclicLR. If None is provided, an error is raised. Default is None.')
parser.add_argument('--step_size_up'            , type = int  , default = None, help = 'Step size up for CyclicLR. If None is provided, an error is raised. Default is None.')
parser.add_argument('--step_size_down'          , type = int  , default = None, help = 'Step size down for CyclicLR. If None is provided, an error is raised. Default is None.')
parser.add_argument('--mode'                    , type = str  , default = 'triangular', help = 'Mode for CyclicLR. Default is "triangular". Other options are "triangular2" and "exp_range".')
parser.add_argument('--lr_scheduler_configs_path_list', nargs='+', default = [], help = 'This argument is used if name is "ChainedScheduler". It is a list of paths to the toml files with the learning rate scheduler configs. If an empty list is provided, an error is raised. Default is an empty list.')
args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Load the toml file
lr_scheduler_config = dict()

valid_name = [
    'ExponentialLR',
    'StepLR',
    'CyclicLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'ChainedScheduler'
]

# Check if the name is provided
if args.name is None:
    raise ValueError('The name of the learning rate scheduler must be provided.')
elif args.name not in valid_name:
    raise ValueError(f'The name of the learning rate scheduler must be one of {valid_name}. Provided: {args.name}')
else :
    lr_scheduler_config['name'] = args.name

# Update the config with the provided parameters
if args.name == 'ExponentialLR': # ExponentialLR
    if args.gamma is None:
        raise ValueError('The gamma parameter must be provided for ExponentialLR.')
    lr_scheduler_config['gamma'] = args.gamma
elif args.name == 'CosineAnnealingLR' : # CosineAnnealingLR
    if args.T_max is None:
        raise ValueError('The T_max parameter must be provided for CosineAnnealingLR.')
    lr_scheduler_config['T_max'] = args.T_max
    lr_scheduler_config['eta_min'] = args.eta_min
elif args.name == 'CosineAnnealingWarmRestarts':
    if args.T_0 is None:
        raise ValueError('The T_0 parameter must be provided for CosineAnnealingWarmRestarts.')
    lr_scheduler_config['T_0'] = args.T_0
    lr_scheduler_config['T_mult'] = args.T_mult
    lr_scheduler_config['eta_min'] = args.eta_min
elif args.name == 'StepLR':
    if args.step_size is None:
        raise ValueError('The step_size parameter must be provided for StepLR.')
    lr_scheduler_config['step_size'] = args.step_size
    if args.gamma is not None:
        lr_scheduler_config['gamma'] = args.gamma
    else :
        lr_scheduler_config['gamma'] = 0.1
elif args.name == 'CyclicLR':
    if args.base_lr is None:
        raise ValueError('The base_lr parameter must be provided for CyclicLR.')
    if args.max_lr is None:
        raise ValueError('The max_lr parameter must be provided for CyclicLR.')
    if args.step_size_up is None:
        raise ValueError('The step_size_up parameter must be provided for CyclicLR.')
    if args.step_size_down is None:
        raise ValueError('The step_size_down parameter must be provided for CyclicLR.')
    lr_scheduler_config['base_lr'] = args.base_lr
    lr_scheduler_config['max_lr'] = args.max_lr
    lr_scheduler_config['step_size_up'] = args.step_size_up
    lr_scheduler_config['step_size_down'] = args.step_size_down
    lr_scheduler_config['mode'] = args.mode
    if args.gamma is not None:
        lr_scheduler_config['gamma'] = args.gamma
    else :
        lr_scheduler_config['gamma'] = 1.0
elif args.name == 'ChainedScheduler' :
    if len(args.lr_scheduler_configs_path_list) == 0:
        raise ValueError('The lr_scheduler_configs_path_list parameter must be provided for ChainedScheduler.')

    list_config_schedulers = dict()
    for i in range(len(args.lr_scheduler_configs_path_list)):
        # Load the toml file
        try:
            lr_scheduler_config_i = toml.load(args.lr_scheduler_configs_path_list[i])
        except FileNotFoundError:
            raise FileNotFoundError(f'The file {args.lr_scheduler_configs_path_list[i]} does not exist.')
        
        # Check if the name is valid
        if lr_scheduler_config_i['name'] not in valid_name:
            raise ValueError(f'The name of the learning rate scheduler must be one of {valid_name}. Provided: {lr_scheduler_config_i["name"]}')

        # Add the config to the list
        list_config_schedulers[f"config_{i}"] = lr_scheduler_config_i

    # Add the list of configs to the main config
    lr_scheduler_config['list_config_schedulers'] = list_config_schedulers

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the config

# Create the folder if it does not exist
os.makedirs(os.path.dirname(args.path_lr_scheduler_config), exist_ok = True)

# Save the config to a toml file
with open(args.path_lr_scheduler_config, 'w') as f:
    toml.dump(lr_scheduler_config, f)

print("Update LR SCHEDULER config - OK")
