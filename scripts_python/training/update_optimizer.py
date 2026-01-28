"""
Update the optimizer config toml file
Note that this script works a little bit differently than uupdate_training_config.py
In the latter, if a config files already exists, it is updated the existing file with the new parameters (if it does not exist of cousre a new toml file is created).
This script instead, always creates a new toml file with the new parameters. If a toml file with the same name already exists, it is overwritten.
The optimizer currently available are Adam, AdamW, LBFGS, SGD. For more information see the function get_optimizer() in support_training.py.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import argparse

# Create parser
parser = argparse.ArgumentParser(description = 'Update the training configuration file with new parameters.')

# *******************************
# General parameters
parser.add_argument('--path_optimizer_config', type = str  , default = './config/optimizer.toml', help = 'Path to the toml file with the optimizer scheduler config. Default is ./config/optimizer_scheduler.toml')
parser.add_argument('--path_src'             , type = str, default = None, help = 'Path of the src folder') # This was added only to execute without problem the line with args.path_src
parser.add_argument('--name'                 , type = str  , default = None, help = 'Name of the optimizer. If None is provided, an error is raised. Default is None.')
parser.add_argument('--lr'                   , type = float, default = 0.001                    , help = 'Learning rate for the optimizer. Default is 0.001 (1e-3).')
# *******************************
# Adam and AdamW parameters
parser.add_argument('--betas'                    , type = float, nargs = '+', default = [0.9, 0.999]   , help = 'Betas for the Adam or AdamW optimizer. Default is [0.9, 0.999].')
parser.add_argument('--eps'                      , type = float             , default = 1e-8           , help = 'Epsilon for the Adam or AdamW optimizer. Default is 1e-08.')
parser.add_argument('--weight_decay'             , type = float             , default = None           , help = 'Weight decay for the Adam, AdamW or SGD optimizer. If None the default value for those optimizer will be used, i.e. 0.0 for SGD and Adam and 1e-2 for AdamW. Default is None')
parser.add_argument('--amsgrad'                  , action = 'store_true'    , default = False          , help = 'Amsgrad for the Adam or AdamW optimizer. Default is False.')
parser.add_argument('--maximize'                 , action = 'store_true'    , default = False          , help = 'Maximize for the Adam or AdamW or SGD optimizer. Default is False.')
parser.add_argument('--decoupled_weight_decay'   , action = 'store_true'    , default = False          , help = 'Decoupled weight decay for the Adam optimizer. Note that if this parameter is set to True, the Adam optimizer will be equivalent to AdamW. Default is False.')
parser.add_argument('--no-amsgrad'               , action = 'store_false'   , dest ='wandb_training')
parser.add_argument('--no-maximize'              , action = 'store_false'   , dest ='wandb_training')
parser.add_argument('--no-decoupled_weight_decay', action = 'store_false'   , dest ='wandb_training')
# *******************************
# LBFGS parameters
parser.add_argument('--max_iter'              , type = int  , default = 20  , help = 'Maximum number of iterations for the LBFGS optimizer. Default is 20.')
parser.add_argument('--max_eval'              , type = int  , default = None, help = 'Maximum number of evaluations for the LBFGS optimizer. If not provided (None), it is set to max_iter * 1.25. Default is None.')
parser.add_argument('--tolerance_grad'        , type = float, default = 1e-7, help = 'Tolerance for the gradient for the LBFGS optimizer. Default is 1e-07.')
parser.add_argument('--tolerance_change'      , type = float, default = 1e-9, help = 'Tolerance for the change in the parameters for the LBFGS optimizer. Default is 1e-09.')
parser.add_argument('--history_size'          , type = int  , default = 100 , help = 'History size for the LBFGS optimizer. Default is 100.')
parser.add_argument('--line_search_fn'        , type = str  , default = None, help = 'Line search function for the LBFGS optimizer. The possible values are "strong_wolfe" and None. Default is None.')
# *******************************
# SGD parameters
parser.add_argument('--momentum'              , type = float, default = 0.0           , help = 'Momentum for the SGD optimizer. Default is 0.0.')
parser.add_argument('--dampening'             , type = float, default = 0.0           , help = 'Dampening for the SGD optimizer. Default is 0.0.')
parser.add_argument('--nesterov'              , default = False, action = 'store_true', help = 'Nesterov for the SGD optimizer. Default is False.')
# *******************************
args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# This specific import was added to allow the execution of the script with the "python" command from any folder you like.
# If the argument path_src is not provided, the script assume you will run it from the root folder of the repository.
import sys
if args.path_src is not None : sys.path.append(args.path_src)
else : sys.path.append('./')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import toml

from src.training.support_training import check_optimizer_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

optimizer_config = dict()

valid_name_list = [
    'Adam',
    'AdamW',
    'LBFGS',
    'SGD'
]

# Check if the name provided is valid
if args.name is None                : raise ValueError('The name of the optimizer must be provided.')
if args.name not in valid_name_list : raise ValueError(f'The name of the optimizer must be one of {valid_name_list}. Provided: {args.name}')

# Set the learning rate
optimizer_config['lr'] = args.lr

# Set parameters in the config
if args.name == 'Adam' or args.name == 'AdamW' :
    optimizer_config['name'] = args.name
    optimizer_config['betas'] = args.betas
    optimizer_config['eps'] = args.eps
    if args.name == 'Adam'  : optimizer_config['weight_decay'] = args.weight_decay if args.weight_decay is not None else 0.0
    if args.name == 'AdamW' : optimizer_config['weight_decay'] = args.weight_decay if args.weight_decay is not None else 1e-2
    optimizer_config['amsgrad'] = args.amsgrad
    optimizer_config['maximize'] = args.maximize
    optimizer_config['decoupled_weight_decay'] = args.decoupled_weight_decay if args.name == 'Adam' else None
elif args.name == 'LBFGS' :
    optimizer_config['name'] = args.name
    optimizer_config['max_iter'] = args.max_iter
    optimizer_config['max_eval'] = args.max_eval if args.max_eval is not None else args.max_iter * 1.25
    optimizer_config['tolerance_grad'] = args.tolerance_grad
    optimizer_config['tolerance_change'] = args.tolerance_change
    optimizer_config['history_size'] = args.history_size
    optimizer_config['line_search_fn'] = args.line_search_fn
elif args.name == 'SGD' :
    optimizer_config['name'] = args.name
    optimizer_config['momentum'] = args.momentum
    optimizer_config['dampening'] = args.dampening
    optimizer_config['nesterov'] = args.nesterov
    optimizer_config['weight_decay'] = args.weight_decay if args.weight_decay is not None else 0.0
    optimizer_config['maximize'] = args.maximize

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check config
check_optimizer_config(optimizer_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the config

# Create the folder if it does not exist
os.makedirs(os.path.dirname(args.path_optimizer_config), exist_ok = True)

# Save the config to a toml file
with open(args.path_optimizer_config, 'w') as f:
    toml.dump(optimizer_config, f)

print("Update OPTIMIZER config - OK")
