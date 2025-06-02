"""
Update the dement model configuration file with new parameters.
For more information about the demnet model see the dement class in src

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

# Non boolean arguments
parser.add_argument('--path_model_config' , type = str  , default = './config/demnet_model.toml'  , help = 'Path to the toml file with the demnet model config. Default is ./config/demnet_model.toml')
parser.add_argument('--input_size'        , type = int  , default = None                          , help = 'Input size of the model. Default is 176.')
parser.add_argument('--input_channels'    , type = int  , default = None                          , help = 'Number of input channels. Default is 1.')
parser.add_argument('--num_classes'       , type = int  , default = None                          , help = 'Number of output classes. Default is 4.')
parser.add_argument('--activation'        , type = str  , default = 'elu'                         , help = 'Activation function to use. Default is "elu".')
parser.add_argument('--kernel_size_conv_1', type = int  , default = 7                             , help = 'Kernel size for the first convolutional layer. Default is 7.')
parser.add_argument('--kernel_size_conv_2', type = int  , default = 5                             , help = 'Kernel size for the second convolutional layer. Default is 5.')
parser.add_argument('--dropout_rate_1'    , type = float, default = 0.4                           , help = 'Dropout rate for the first dropout layer. Default is 0.4.')
parser.add_argument('--dropout_rate_2'    , type = float, default = 0.4                           , help = 'Dropout rate for the second dropout layer. Default is 0.4.')
# List to dement block config files
parser.add_argument('--demnet_blocks_path_list'     , nargs = '+', default = [], help = 'List of paths to the toml files with the dement block model config. If not provided, an error will be raised. Default is an empty list. Example: --demnet_blocks ./config/demnet_block_1.toml ./config/demnet_block_2.toml')# Boolean arguments
# Boolean arguments
parser.add_argument('--batch_norm'                  , default = True , action = 'store_true', help = 'If set, batch normalization will be used. Default is True.')
parser.add_argument('--use_activation_in_classifier', default = True , action = 'store_true', help = 'If set, the classifier will use an activation function. Default is True.')
parser.add_argument('--use_as_features_extractor'   , default = False, action = 'store_true', help = 'If set, the model will be used as a feature extractor. Default is False.')
# Negative boolean arguments
parser.add_argument('--no_batch_norm'                  , action = 'store_false', dest = 'batch_norm'                  , help = 'If set, batch normalization will not be used. Default is False.')
parser.add_argument('--no_use_activation_in_classifier', action = 'store_false', dest = 'use_activation_in_classifier', help = 'If set, the classifier will not use an activation function. Default is False.')
parser.add_argument('--no_use_as_features_extractor'   , action = 'store_false', dest = 'use_as_features_extractor'   , help = 'If set, the model will not be used as a feature extractor. Default is False.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

model_config = {}

if args.input_size is not None:
    if args.input_size <= 0:
        raise ValueError(f'Input size must be a positive integer. Given: {args.input_size}')
    else :
        model_config['input_size'] = args.input_size
else :
    raise ValueError('Input size must be specified. Use --input_size to set it.')

if args.input_channels is not None: 
    if args.input_channels <= 0:
        raise ValueError(f'Input channels must be a positive integer. Given: {args.input_channels}')
    else :
        model_config['input_channels'] = args.input_channels
else : 
    raise ValueError('Input channels must be specified. Use --input_channels to set it.')

if args.num_classes is not None:
    if args.num_classes <= 0:
        raise ValueError(f'Number of classes must be a positive integer. Given: {args.num_classes}')
    else :
        model_config['num_classes'] = args.num_classes
else :
    raise ValueError('Number of classes must be specified. Use --num_classes to set it.')

allowed_activations = ['elu', 'relu', 'gelu', 'selu']
if args.activation in allowed_activations:
    model_config['activation'] = args.activation
else :
    raise ValueError(f'Activation function must be one of {allowed_activations}. Given: {args.activation}')

if args.kernel_size_conv_1 is not None:
    if args.kernel_size_conv_1 <= 0:
        raise ValueError(f'Kernel size for the first convolutional layer must be a positive integer. Given: {args.kernel_size_conv_1}')
    else :
        model_config['kernel_size_conv_1'] = args.kernel_size_conv_1
else :
    print('Kernel size for the first convolutional layer not specified. Using default value of 7.')
    model_config['kernel_size_conv_1'] = 7

if args.kernel_size_conv_2 is not None: 
    if args.kernel_size_conv_2 <= 0:
        raise ValueError(f'Kernel size for the second convolutional layer must be a positive integer. Given: {args.kernel_size_conv_2}')
    else :
        model_config['kernel_size_conv_2'] = args.kernel_size_conv_2
else :
    print('Kernel size for the second convolutional layer not specified. Using default value of 5.')
    model_config['kernel_size_conv_2'] = 5

if args.dropout_rate_1 is not None: 
    if args.dropout_rate_1 < 0 or args.dropout_rate_1 > 1:
        raise ValueError(f'Dropout rate for the first dropout layer must be between 0 and 1. Given: {args.dropout_rate_1}')
    else :
        model_config['dropout_rate_1'] = args.dropout_rate_1

if args.dropout_rate_2 is not None:
    if args.dropout_rate_2 < 0 or args.dropout_rate_2 > 1:
        raise ValueError(f'Dropout rate for the second dropout layer must be between 0 and 1. Given: {args.dropout_rate_2}')
    else :
        model_config['dropout_rate_2'] = args.dropout_rate_2

if len(args.demnet_blocks_path_list) > 0 :
    for i in range(len(args.demnet_blocks_path_list)):
        if not args.demnet_blocks_path_list[i].endswith('.toml'):
            raise ValueError(f'Path to the dement block config file must end with .toml. Given: {args.demnet_blocks_path_list[i]}')
        else :
            tmp_block_config = toml.load(args.demnet_blocks_path_list[i])
            model_config[f'config_demnet_block_{i + 1}'] = tmp_block_config
else :
    raise ValueError('At least one path to the dement block config file must be specified. Use --demnet_blocks_path_list to set it.')

