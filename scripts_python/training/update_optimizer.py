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
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the training configuration file with new parameters.')

# Add arguments
parser.add_argument('--path_optimizer_config', type = str  , default = './config/optimizer_scheduler.toml', help = 'Path to the toml file with the optimizer scheduler config. Default is ./config/optimizer_scheduler.toml')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

optimizer_config = dict()

