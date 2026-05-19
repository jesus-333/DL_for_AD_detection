"""
Update/create the vav model configuration file with new parameters.
See the class in vav.py for more details about the model and the config parameters.

If you provide a template file, the new config file will be created starting from the template. 
In this case, only the parameters passed as arguments will be updated, while the other parameters will be kept unchanged.
The file will be saved in the path specified by the --path_save argument. If the file already exists, it will be overwritten.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import os
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the ViT model configuration file with new parameters.')

parser.add_argument('--path_save'                    , type = str  , default = './config/vit_model.toml', help = 'Path to save the updated model config file. Default is ./config/vit_model.toml')
parser.add_argument('--path_template'                , type = str  , default = None, help = 'Path to the toml file with a template of the model config')
parser.add_argument('--num_classes'                  , type = int  , default = None, help = 'Number of output classes. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--path_vit_config'              , type = str  , default = None, help = 'Path to the toml file with the ViT config. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--path_slice_transformer_config', type = str  , default = None, help = 'Path to the toml file with the slice transformer config. If None is passed, the value already present in the config file will be used. Default is None.')
# Boolean argument
parser.add_argument('--use_same_vit_for_all_directions' , default = None, action = 'store_true', help = 'If passed, the same ViT will be used for all three directions. If not passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--analyze_directions_independently', default = None, action = 'store_true', help = 'If passed, a separate transformer encoder will be used for each direction to process the sequence of embeddings produced by the ViT for that direction. If not passed, the value already present in the config file will be used. Default is None.')
# Negative boolean arguments
parser.add_argument('--no-use_same_vit_for_all_directions' , action = 'store_false', dest = 'use_same_vit_for_all_directions' , help = 'If passed, a separate ViT will be used for each direction. If not passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--no-analyze_directions_independently', action = 'store_false', dest = 'analyze_directions_independently', help = 'If passed, a single transformer encoder will be used to process the concatenation of the three sequences of embeddings produced by the ViTs for all three directions. If not passed, the value already present in the config file will be used. Default is None.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_int_input(model_config : dict, parameter_name : str, parameter_value : int, threshold : int) -> None :
    """
    Function to check the value of an integer parameter and update the model config dictionary accordingly.

    Parameters
    ----------
    model_config : dict
        The model config dictionary to update.
    parameter_name : str
        The name of the parameter to check and update.
    parameter_value : int
        The value of the parameter to check. If None is passed, the value already present in the config file will be used.
    threshold : int
        The threshold value that the parameter value must be greater than. If the parameter value is not greater than the threshold, a ValueError is raised.
    """
    if parameter_value is None : # Parameter not provided, check if it is present in the config file
        if parameter_name not in model_config :
            raise ValueError(f'{parameter_name} not provided and not present in the config file. Please provide a value for {parameter_name}.')
        else :
            print(f"{parameter_name} not provided. The value already present in the config file will be used. Current value is {model_config[parameter_name]}")
    else : # Parameter provided, check if it is valid and update the config file
        if parameter_value > threshold :
            model_config[parameter_name] = parameter_value
        else :
            raise ValueError(f'The value of {parameter_name} must be > {threshold}. Current value is {parameter_value}')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Check if a template file is provided and exists. If it exists, load the model config from the template file. If it does not exist, create a new model config dictionary.
if os.path.exists(args.path_template) :
    print("Using the template file to create a new model config.")
    model_config = toml.load(args.path_template)
else :
    if args.path_template is not None :
        print(f'Template provided but the file {args.path_template} does not exist. A new file will be created.')
    else :
        print('No template file provided. A new file will be created.')

    model_config = {}

# Number of classes
check_int_input(model_config, 'num_classes', args.num_classes, 0)

# ViT config
if args.path_vit_config is None :
    if 'path_vit_config' not in model_config :
        raise ValueError('path_vit_config not provided and not present in the config file. Please provide a value for path_vit_config.')
    else :
        print(f"path_vit_config not provided. The value already present in the config file will be used. Current value is {model_config['path_vit_config']}")
else :
    if not args.path_vit_config.endswith('.toml') :
        raise ValueError('path_vit_config must be a toml file. Current value is {args.path_vit_config}')
    elif not os.path.exists(args.path_vit_config) :
        raise ValueError(f'The file {args.path_vit_config} does not exist. Please provide a valid path for path_vit_config.')
    else :
        model_config['vit_config'] = toml.load(args.path_vit_config)

    # TODO ADD ViT config check


# Slice transformer config
if args.path_slice_transformer_config is None :
    if 'path_slice_transformer_config' not in model_config :
        raise ValueError('path_slice_transformer_config not provided and not present in the config file. Please provide a value for path_slice_transformer_config.')
    else :
        print(f"path_slice_transformer_config not provided. The value already present in the config file will be used. Current value is {model_config['path_slice_transformer_config']}")
else :
    if not args.path_slice_transformer_config.endswith('.toml') :
        raise ValueError('path_slice_transformer_config must be a toml file. Current value is {args.path_slice_transformer_config}')
    elif not os.path.exists(args.path_slice_transformer_config) :
        raise ValueError(f'The file {args.path_slice_transformer_config} does not exist. Please provide a valid path for path_slice_transformer_config.')
    else :
        model_config['slice_transformer_config'] = toml.load(args.path_slice_transformer_config)

    # TODO ADD slice transformer config check

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save boolean arguments
# Pretrained
if args.pretrained is None :
    if 'pretrained' not in model_config :
        raise ValueError('pretrained not provided and not present in the config file. Please provide a value for pretrained.')
    else :
        print(f"pretrained not provided. The value already present in the config file will be used. Current value is {model_config['pretrained']}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the config

# Create the folder if it does not exist
os.makedirs(os.path.dirname(args.path_save), exist_ok = True)

# Save the updated config
with open(args.path_save, 'w') as f :
    toml.dump(model_config, f)

print("Update MODEL config (vav) - OK")
