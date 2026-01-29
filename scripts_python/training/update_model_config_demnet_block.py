"""
Update the demnet block config filg

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)

CURRENTLY NOT USED
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import numpy as np
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the demnet block model configuration file with new parameters.')

# Add arguments

# Various arguments
parser.add_argument('--path_model_config'    , type = str, default = './config/demnet_block_model.toml' , help = 'Path to the toml file with the demnet block model config. Default is ./config/demnet_block_model.toml')
parser.add_argument('--input_channels_block' , type = int, default = 16                                 , help = 'Number of input channels for the block. Default is 16.')
parser.add_argument('--output_channels_block', type = int, default = 32                                 , help = 'Number of output channels for the block. Default is 32.')
parser.add_argument('--kernel_size_conv_1'   , type = int, default = 3                                  , help = 'Kernel size for the first convolutional layer in the block. Default is 3.')
parser.add_argument('--kernel_size_conv_2'   , type = int, default = 3                                  , help = 'Kernel size for the second convolutional layer in the block. Default is 3.')
parser.add_argument('--activation'           , type = str, default = 'elu'                              , help = 'Activation function to use in the block. Default is "elu".')
parser.add_argument('--batch_norm'           , default = True , action = 'store_true', help = 'If set, batch normalization will be used in the block. Default is True.')
# ADD no-batch norm option


