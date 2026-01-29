"""
Update the dement model configuration file with new parameters.

NOTES ABOUT PATH
Originally, as the update_training_config.py script, this script did not overwrite the file, but only updated it. There was a single parameter, called path_model_config, that specified the path to the config file to update.
Of course if no file was present at that path, a new one was created.

Later I noticed that with this approach if I launched multiple training with different parameters, I risk to create conflicts between them, if the same path_model_config was used.
Why you should use the same path for multiple training? Because when I launch the training with the sh script I not update all parameters of the model but only a few of them.
Exploiting the fact that this script can overwrite only the parameters that I want to change, allow me to avoid to write too much settings in the sh script (like all the demnet block config that remains always the same).
But as I said, this could create conflicts between multiple training run. To avoid this problem, I added a new parameter, called path_template and renamed path_model_config to path_save.
In this way I use the file from path_template as a bae to create a new config file that is saved at path_save. All the parameters that are not specified in the command line remain the same as in the template file. The new parameters specified in the command line overwrite the old ones and the new config file is saved at path_save.

I will probably add this feature also to the other update_xxx_config.py scripts.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import os
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the demnet model configuration file with new parameters.')

# Non boolean arguments
parser.add_argument('--path_save'         , type = str  , default = './config/demnet_model.toml', help = 'Path to save the updated model config file. Default is ./config/demnet_model.toml')
parser.add_argument('--path_template'     , type = str  , default = None, help = 'Path to the toml file with a template of the model config')
parser.add_argument('--input_size'        , type = int  , default = None, help = 'Input size of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--input_channels'    , type = int  , default = None, help = 'Number of input channels. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--num_classes'       , type = int  , default = None, help = 'Number of output classes. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--activation'        , type = str  , default = None, help = 'Activation function to use. If None is passed, the value already present in the config file will be used. Default is None. Allowed values are: elu, relu, gelu, selu.')
parser.add_argument('--kernel_size_conv_1', type = int  , default = None, help = 'Kernel size for the first convolutional layer. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--kernel_size_conv_2', type = int  , default = None, help = 'Kernel size for the second convolutional layer. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--dropout_rate_1'    , type = float, default = None, help = 'Dropout rate for the first dropout layer. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--dropout_rate_2'    , type = float, default = None, help = 'Dropout rate for the second dropout layer. If None is passed, the value already present in the config file will be used. Default is None.')
# List to dement block config files
parser.add_argument('--demnet_blocks_path_list'     , nargs = '+', default = [], help = 'List of paths to the toml files with the dement block model config. Default is an empty list. Example: --demnet_blocks ./config/demnet_block_1.toml ./config/demnet_block_2.toml')
# Boolean arguments
parser.add_argument('--batch_norm'                  , default = None, action = 'store_true', help = 'If passed, batch normalization will be used. Default is True.')
parser.add_argument('--use_activation_in_classifier', default = None, action = 'store_true', help = 'If passed, the classifier will use an activation function. Default is True.')
parser.add_argument('--use_as_features_extractor'   , default = None, action = 'store_true', help = 'If passed, the model will be used as a feature extractor. Default is False.')
# Negative boolean arguments
parser.add_argument('--no-batch_norm'                  , action = 'store_false', dest = 'batch_norm'                  , help = 'If passed, batch normalization will not be used.')
parser.add_argument('--no-use_activation_in_classifier', action = 'store_false', dest = 'use_activation_in_classifier', help = 'If passed, the classifier will not use an activation function.')
parser.add_argument('--no-use_as_features_extractor'   , action = 'store_false', dest = 'use_as_features_extractor'   , help = 'If passed, the model will not be used as a feature extractor.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Check if a template file is provided
if os.path.exists(args.path_template) :
    print("Using the template file to create a new model config.")
    model_config = toml.load(args.path_template)
else :
    if args.path_template is not None :
        print(f'Template provided but the file {args.path_template} does not exist. A new file will be created.')
    else :
        print('No template file provided. A new file will be created.')

    model_config = {}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save NON boolean arguments

if args.input_size is not None:
    if args.input_size <= 0:
        raise ValueError(f'Input size must be a positive integer. Given: {args.input_size}')
    else :
        model_config['input_size'] = args.input_size

if args.input_channels is not None:
    if args.input_channels <= 0:
        raise ValueError(f'Input channels must be a positive integer. Given: {args.input_channels}')
    else :
        model_config['input_channels'] = args.input_channels

if args.num_classes is not None:
    if args.num_classes <= 0:
        raise ValueError(f'Number of classes must be a positive integer. Given: {args.num_classes}')
    else :
        model_config['num_classes'] = args.num_classes

if args.activation is not None :
    allowed_activations = ['elu', 'relu', 'gelu', 'selu']
    if args.activation in allowed_activations:
        model_config['activation'] = args.activation
    else :
        raise ValueError(f'Activation function must be one of {allowed_activations}. Given: {args.activation}')

if args.kernel_size_conv_1 is not None :
    if args.kernel_size_conv_1 <= 0:
        raise ValueError(f'Kernel size for the first convolutional layer must be a positive integer. Given: {args.kernel_size_conv_1}')
    else :
        model_config['kernel_size_conv_1'] = args.kernel_size_conv_1

if args.kernel_size_conv_2 is not None :
    if args.kernel_size_conv_2 <= 0:
        raise ValueError(f'Kernel size for the second convolutional layer must be a positive integer. Given: {args.kernel_size_conv_2}')
    else :
        model_config['kernel_size_conv_2'] = args.kernel_size_conv_2

if args.dropout_rate_1 is not None :
    if args.dropout_rate_1 < 0 or args.dropout_rate_1 > 1:
        raise ValueError(f'Dropout rate for the first dropout layer must be between 0 and 1. Given: {args.dropout_rate_1}')
    else :
        model_config['dropout_rate_1'] = args.dropout_rate_1

if args.dropout_rate_2 is not None :
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
# else :
#     raise ValueError('At least one path to the dement block config file must be specified. Use --demnet_blocks_path_list to set it.')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save boolean arguments
if args.batch_norm is not None : model_config['batch_norm'] = args.batch_norm
if args.use_activation_in_classifier is not None : model_config['use_activation_in_classifier'] = args.use_activation_in_classifier
if args.use_as_features_extractor is not None : model_config['use_as_features_extractor'] = args.use_as_features_extractor

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the config

# Create the folder if it does not exist
os.makedirs(os.path.dirname(args.path_save), exist_ok = True)

# Save the updated config
with open(args.path_save, 'w') as f :
    toml.dump(model_config, f)

print("Update MODEL config (DEMNET) - OK")
