"""
Update the server config toml file for the FL training simulation with the fed_avg_with_wandb_tracking class.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import numpy as np
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the server configuration file with new parameters.')

# Add arguments

# Various arguments
parser.add_argument('--path_pyproject', type = str  , default = None, help = 'Path to the toml file with the server config. Default is ./config/serve.toml')
args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

pyproject_config = toml.load(args.path_pyproject)

# {'tool': {'flwr': {'app': {'publisher': 'Alberto (Jesus) Zancanaro',
#     'components': {'serverapp': 'server_app:app',
#      'clientapp': 'client_app:app'}},
#    'federations': {'default': 'local-simulation',
#     'local-simulation': {'options': {'num-supernodes': 4}}}}}}

