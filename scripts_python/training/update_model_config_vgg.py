"""
Update the vgg model configuration file with new parameters.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import os
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the vgg model configuration file with new parameters.')

# Non boolean arguments
parser.add_argument('--path_save'         , type = str  , default = './config/vgg_model.toml', help = 'Path to save the updated model config file. Default is ./config/vgg_model.toml')
parser.add_argument('--path_template'     , type = str  , default = None, help = 'Path to the toml file with a template of the model config')
parser.add_argument('--input_channels'    , type = int  , default = None, help = 'Number of input channels. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--num_classes'       , type = int  , default = None, help = 'Number of output classes. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--version'           , type = int  , default = None, help = 'Version of the VGG network. Possible values are 11, 13, 16 or 19. Default is None.')
# Boolean arguments
parser.add_argument('--batch_norm'        , default = True, action = 'store_true', help = 'If passed, batch normalization will be used. Default is True.')
parser.add_argument('--use_pretrained_vgg', default = True, action = 'store_true', help = 'If passed, the VGG model will inizialized with the weights provided by torch hub. Default is True.')
# Negative boolean arguments
parser.add_argument('--no-batch_norm'        , action = 'store_false', dest = 'batch_norm', help = 'If passed, batch normalization will not be used.')
parser.add_argument('--no-use_pretrained_vgg', action = 'store_false', dest = 'batch_norm', help = 'If passed, the VGG model will be randomly initialized.')

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

if args.version is not None:
    if args.version not in [11, 13, 16, 19] :
        raise ValueError(f'Version must be 11, 13, 16 or 19. Given: {args.version}')
    else :
        model_config['version'] = args.version

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save boolean arguments
if args.batch_norm is not None : model_config['batch_norm'] = args.batch_norm
if args.use_pretrained_vgg is not None : model_config['use_pretrained_vgg'] = args.use_pretrained_vgg

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the config

# Create the folder if it does not exist
os.makedirs(os.path.dirname(args.path_save), exist_ok = True)

# Save the updated config
with open(args.path_save, 'w') as f :
    toml.dump(model_config, f)

print("Update MODEL config (VGG) - OK")

