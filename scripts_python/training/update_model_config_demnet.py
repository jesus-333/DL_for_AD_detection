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
parser = argparse.ArgumentParser(description = 'Update the demnet model configuration file with new parameters.')

# Add arguments

# Various arguments
parser.add_argument('--path_model_config' , type = str  , default = './config/demnet_model.toml'  , help = 'Path to the toml file with the demnet model config. Default is ./config/demnet_model.toml')
parser.add_argument('--input_size'        , type = int  , default = 176                           , help = 'Input size of the model. Default is 176.')
parser.add_argument('--input_channels'    , type = int  , default = 1                             , help = 'Number of input channels. Default is 1.')
parser.add_argument('--num_classes'       , type = int  , default = 4                             , help = 'Number of output classes. Default is 4.')
parser.add_argument('--activation'        , type = str  , default = 'elu'                         , help = 'Activation function to use. Default is "elu".')
parser.add_argument('--kernel_size_conv_1', type = int  , default = 7                             , help = 'Kernel size for the first convolutional layer. Default is 7.')
parser.add_argument('--kernel_size_conv_2', type = int  , default = 5                             , help = 'Kernel size for the second convolutional layer. Default is 5.')
parser.add_argument('--dropout_rate_1'    , type = float, default = 0.4                           , help = 'Dropout rate for the first dropout layer. Default is 0.4.')
parser.add_argument('--dropout_rate_2'    , type = float, default = 0.4                           , help = 'Dropout rate for the second dropout layer. Default is 0.4.')
parser.add_argument('--batch_norm'                  , default = True , action = 'store_true', help = 'If set, batch normalization will be used. Default is True.')
parser.add_argument('--use_activation_in_classifier', default = True , action = 'store_true', help = 'If set, the classifier will use an activation function. Default is True.')
parser.add_argument('--use_as_features_extractor'   , default = False, action = 'store_true', help = 'If set, the model will be used as a feature extractor. Default is False.')

# ADD negative bool options
# Add list to dement block
