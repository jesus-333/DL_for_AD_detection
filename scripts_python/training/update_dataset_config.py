"""
Update the dataset config toml file

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the training configuration file with new parameters.')

# Non-boolean arguments
parser.add_argument('--path_dataset_config'  , type = str  , default = './config/dataset.toml', help = 'Path to the toml file with the learning rate scheduler config. Default is ./config/dataset.toml')
parser.add_argument('--path_data'            , type = str  , default = './data/'              , help = 'Path to the folder with the data. Note that this is not the path to a file (as in the case of the training config), but the path to the folder where the data is stored. Default is ./data/.')
parser.add_argument('--name_tensor_file'     , type = str  , default = 'dataset_tensor.pt'    , help = 'Name of the tensor file with the dataset. This is usefull only if the data are stored in a single tensor file. Default is dataset_tensor.pt.')
parser.add_argument('--path_idx_folder'      , type = str  , default = None                   , help = 'Path to the folder where the indices are stored. Used only in FL. Default is None.')
parser.add_argument('--merge_AD_class'       , type = int  , default = 0                      , help = 'Merge the AD class with the CN class. Default is 0 (do not merge). For the other options see the training script.')
parser.add_argument('--percentage_train'     , type = float, default = 0.7                    , help = 'Percentage of the dataset to use for training. Default is 0.7.')
parser.add_argument('--percentage_validation', type = float, default = 0.15                   , help = 'Percentage of the dataset to use for validation. Default is 0.15.')
parser.add_argument('--percentage_test'      , type = float, default = 0.15                   , help = 'Percentage of the dataset to use for testing. Default is 0.15.')
parser.add_argument('--rescale_factor'       , type = float, default = 4095                   , help = 'Value used in the division if apply_rescale is passed as argument. Default is 4095')
# Boolean arguments
parser.add_argument('--apply_rescale'       , default = False, action = 'store_true', help = 'Rescale the value of the data dividing by the value stored in rescale_factor. This is used in case the data where stored in uint16 (or similar format) and you want to rescale them to the 0-1 range. Default is False')
parser.add_argument('--use_normalization'   , default = True , action = 'store_true', help = 'Use normalization in the dataset. Default is True.')
parser.add_argument('--filter_AD_data'      , default = False, action = 'store_true', help = 'Filter the AD data from the dataset. Only the class passed in the class_to_keep argument will be kept. Default is False. If True, the class_to_keep argument must be passed.')
parser.add_argument('--load_data_in_memory' , default = False, action = 'store_true', help = 'Load the data in memory. Default is False.')
parser.add_argument('--use_rgb_input'       , default = None , action = 'store_true', help = 'Use RGB input instead of grayscale. This is used only if the model requires RGB input (e.g. VGG).')
# Boolean negate
parser.add_argument('--no-apply_rescale'      , dest = 'apply_rescale'      , action = 'store_false')
parser.add_argument('--no-use_normalization'  , dest = 'use_normalization'  , action = 'store_false')
parser.add_argument('--no-load_data_in_memory', dest = 'load_data_in_memory', action = 'store_false')
parser.add_argument('--no-filter_AD_data'     , dest = 'filter_AD_data'     , action = 'store_false')
parser.add_argument('--no-use_rgb_input'      , dest = 'use_rgb_input'      , action = 'store_false')
# nargs
parser.add_argument('--class_to_keep', nargs = '+', default = [], help = 'Class to keep when filtering the AD data. Used only if filter_AD_data is set to True. Default is an empty list. You could pass the argument as a list of strings or a list of integers. Example: --class_to_keep CN AD or --class_to_keep 0 1')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Load the dataset config
dataset_config = dict()

dataset_config['path_dataset_config'] = args.path_dataset_config

# Save path to data folder and name of the tensor file
dataset_config['path_data'] = args.path_data
dataset_config['name_tensor_file'] = args.name_tensor_file

# Path to index folder (used only in FL)
if args.path_idx_folder is not None :
    dataset_config['path_idx_folder'] = args.path_idx_folder

# Filter AD data
# Note that no check is done on the class_to_keep values. The check are done later, inside the function filter_AD_data (inside support_dataset_ADNI.py)
dataset_config['filter_AD_data'] = args.filter_AD_data
if args.filter_AD_data :
    if len(args.class_to_keep) == 0 :
        raise ValueError("If filter_AD_data is set to True, class_to_keep must be provided as a list of strings or integers.")
    else :
        dataset_config['class_to_keep'] = args.class_to_keep

# merge_AD_class
if args.merge_AD_class not in [0, 1, 2] :
    raise ValueError(f"merge_AD_class must be 0, 1, or 2. Provided value: {args.merge_AD_class}")
else :
    dataset_config['merge_AD_class'] = args.merge_AD_class

# Check if the percentages are valid
if args.percentage_train < 0.0 or args.percentage_validation < 0.0 or args.percentage_test < 0.0 :
    raise ValueError(f"Percentage values must be non-negative. Provided values: train={args.percentage_train}, validation={args.percentage_validation}, test={args.percentage_test}")
if args.percentage_train > 1.0 or args.percentage_validation > 1.0 or args.percentage_test > 1.0 :
    raise ValueError(f"Percentage values must be less than or equal to 1.0. Provided values: train={args.percentage_train}, validation={args.percentage_validation}, test={args.percentage_test}")

# Check if the sum of the percentages is 1.0
sum_percentages = args.percentage_train + args.percentage_validation + args.percentage_test
possible_sum = [1, 0.9999999999999999, 0.9999999999999998, 1.0000000000000001, 1.0000000000000002]
if sum_percentages in possible_sum :
    # The check with 0.9999999999999999 is to avoid floating point precision issues

    # Save the percentages in the dataset config
    dataset_config['percentage_train']      = args.percentage_train
    dataset_config['percentage_test']       = args.percentage_test
    dataset_config['percentage_validation'] = args.percentage_validation
else :
    raise ValueError(f"The sum of the split percentages must be 1.0. Provided values: train={args.percentage_train}, validation={args.percentage_validation}, test={args.percentage_test}, with sum={sum_percentages}")

# Check recale factor
if args.apply_rescale :
    if args.rescale_factor <= 0 :
        raise ValueError(f"rescale_factor must be a positive number. Provided value: {args.rescale_factor}")
    dataset_config['rescale_factor'] = args.rescale_factor

# Save boolean arguments
dataset_config['apply_rescale']       = args.apply_rescale
dataset_config['load_data_in_memory'] = args.load_data_in_memory
dataset_config['use_normalization']   = args.use_normalization
if args.use_rgb_input is not None : dataset_config['use_rgb_input'] = args.use_rgb_input

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the updated dataset config
with open(args.path_dataset_config, 'w') as f:
    toml.dump(dataset_config, f)


print("Update DATASET config - OK")
