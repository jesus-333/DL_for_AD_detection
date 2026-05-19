"""
Update/create the slice_transformer model configuration file with new parameters.
See the class implementation in vav.py for more details about the model and the config parameters.

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

parser.add_argument('--path_save'       , type = str  , default = './config/vit_model.toml', help = 'Path to save the updated model config file. Default is ./config/vit_model.toml')
parser.add_argument('--path_template'   , type = str  , default = None, help = 'Path to the toml file with a template of the model config')
parser.add_argument('--num_classes'     , type = int  , default = None, help = 'Number of output classes. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--d_model'         , type = int  , default = None, help = 'Dimension of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--nhead'           , type = int  , default = None, help = 'Number of heads of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--dim_feedforward' , type = int  , default = None, help = 'Dimension of the feedforward network of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--dropout'         , type = float, default = None, help = 'Dropout rate of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--activation'      , type = str  , default = None, help = 'Activation function of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--num_layers'      , type = int  , default = None, help = 'Number of layers of the model. If None is passed, the value already present in the config file will be used. Default is None.')
# Boolean argument
# parser.add_argument('--use_same_vit_for_all_directions' , default = None, action = 'store_true', help = 'If passed, the same ViT will be used for all three directions. If not passed, the value already present in the config file will be used. Default is None.')
# Negative boolean arguments
# parser.add_argument('--no-use_same_vit_for_all_directions' , action = 'store_false', dest = 'use_same_vit_for_all_directions' , help = 'If passed, a separate ViT will be used for each direction. If not passed, the value already present in the config file will be used. Default is None.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_numerical_input(model_config : dict, parameter_name : str, parameter_value : int | float, threshold_min : int | float = None, threshold_max : int | float = None) -> None :
    """
    Function to check the value of a numerical parameter and update the model config dictionary accordingly.

    You can specify a minimum and/or maximum threshold for the parameter value. If the parameter value is not within the specified thresholds, a ValueError is raised.
    Note that the thresholds are exclusive, i.e. the parameter value must be greater than the minimum threshold and less than the maximum threshold (if threshold_min = a and threshold_max = b, the parameter value must be a < parameter_value < b).

    Parameters
    ----------
    model_config : dict
        The model config dictionary to update.
    parameter_name : str
        The name of the parameter to check and update.
    parameter_value : int | float
        The value of the parameter to check. If None is passed, the value already present in the config file will be used.
    threshold_min : int | float, optional
        The minimum threshold value that the parameter value must be greater than. If the parameter value is not greater than the minimum threshold, a ValueError is raised. Default is None, i.e. no minimum threshold is applied.
    threshold_max : int | float, optional
        The maximum threshold value that the parameter value must be less than. If the parameter value is not less than the maximum threshold, a ValueError is raised. Default is None, i.e. no maximum threshold is applied.
    """
    if parameter_value is None : # Parameter not provided, check if it is present in the config file
        if parameter_name not in model_config :
            raise ValueError(f'{parameter_name} not provided and not present in the config file. Please provide a value for {parameter_name}.')
        else :
            print(f"{parameter_name} not provided. The value already present in the config file will be used. Current value is {model_config[parameter_name]}")
    else : # Parameter provided, check if it is valid and update the config file
        # Check minimum threshold
        if threshold_min is not None :
            if parameter_value < threshold_min :
                raise ValueError(f'The value of {parameter_name} must be > {threshold_min}. Current value is {parameter_value}')

        # Check maximum threshold
        if threshold_max is not None :
            if parameter_value > threshold_max :
                raise ValueError(f'The value of {parameter_name} must be < {threshold_max}. Current value is {parameter_value}')

        model_config[parameter_name] = parameter_value

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
check_numerical_input(model_config, 'num_classes', args.num_classes, threshold_min = 0)

# Dimension of the model
check_numerical_input(model_config, 'd_model', args.d_model, threshold_min = 0)

# Number of heads
check_numerical_input(model_config, 'nhead', args.nhead, threshold_min = 0)

# Dimension of the feedforward network
check_numerical_input(model_config, 'dim_feedforward', args.dim_feedforward, threshold_min = 0)

# Dropout rate
check_numerical_input(model_config, 'dropout', args.dropout, threshold_min = 0, threshold_max = 1)

# Activation function
if args.activation is None :
    if 'activation' not in model_config :
        raise ValueError('activation not provided and not present in the config file. Please provide a value for activation.')
    else :
        print(f"activation not provided. The value already present in the config file will be used. Current value is {model_config['activation']}")
else :
    valid_activations = ['relu', 'gelu', 'glu']
    if args.activation not in valid_activations :
        raise ValueError(f'Invalid value for activation. Valid values are {valid_activations}. Current value is {args.activation}')
    model_config['activation'] = args.activation

# Number of layers
check_numerical_input(model_config, 'num_layers', args.num_layers, threshold_min = 0)

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

print("Update MODEL config (slice_transformer) - OK")
