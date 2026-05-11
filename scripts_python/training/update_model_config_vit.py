"""
Update/create the ViT model configuration file with new parameters.

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

parser.add_argument('--path_save'                  , type = str  , default = './config/vit_model.toml', help = 'Path to save the updated model config file. Default is ./config/vit_model.toml')
parser.add_argument('--path_template'              , type = str  , default = None, help = 'Path to the toml file with a template of the model config')
parser.add_argument('--model_name'                 , type = str  , default = None, help = 'Name of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--input_size'                 , type = int  , default = None, help = 'Input size of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--num_classes'                , type = int  , default = None, help = 'Number of output classes. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--embed_dim'                  , type = int  , default = None, help = 'Embedding dimension of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--depth'                      , type = int  , default = None, help = 'Depth of the model. If None is passed, the value already present in the config file will be used. Default is None.')
parser.add_argument('--num_heads'                  , type = int  , default = None, help = 'Number of heads of the model. If None is passed, the value already present in the config file will be used. Default is None.')
# Boolean argument
parser.add_argument('--pretrained'                 , default = None, action = 'store_true', help = 'If passed, load the pretrained weights. If not passed, the value already present in the config file will be used. Default is None.')
# Negative boolean arguments
parser.add_argument('--no-pretrained'              , action = 'store_false', dest = 'pretrained', help = 'If passed, do not load the pretrained weights. If not passed, the value already present in the config file will be used. Default is None.')

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


# Model name
if args.model_name is not None :
    model_config['model_name'] = args.model_name
else :
    if 'model_name' not in model_config :
        raise ValueError('model_name not provided and not present in the config file. Please provide a value for model_name.')
    else :
        print(f"model_name not provided. The value already present in the config file will be used. Current value is {model_config['model_name']}")

# Input size
check_int_input(model_config, 'input_size', args.input_size, 0)

# Number of classes
check_int_input(model_config, 'num_classes', args.num_classes, 0)

# Embedding dimension
check_int_input(model_config, 'embed_dim', args.embed_dim, 0)

# Depth
check_int_input(model_config, 'depth', args.depth, 0)

# Number of heads
check_int_input(model_config, 'num_heads', args.num_heads, 0)

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

print("Update MODEL config (ViT) - OK")
