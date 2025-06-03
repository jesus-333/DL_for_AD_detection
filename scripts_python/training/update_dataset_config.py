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
parser.add_argument('--merge_AD_class'       , type = int  , default = 0                      , help = 'Merge the AD class with the CN class. Default is 0 (do not merge). For the other options see the training script.')
parser.add_argument('--percentage_train'     , type = float, default = 0.7                    , help = 'Percentage of the dataset to use for training. Default is 0.7.')
parser.add_argument('--percentage_validation', type = float, default = 0.15                   , help = 'Percentage of the dataset to use for validation. Default is 0.15.')
parser.add_argument('--percentage_test'      , type = float, default = 0.15                   , help = 'Percentage of the dataset to use for testing. Default is 0.15.')
parser.add_argument('--rescale_factor'       , type = float, default = 4095                   , help = 'Value used in the division if apply_rescale is passed as argument. Default is 4095')
# Boolean arguments
parser.add_argument('--apply_rescale'       , default = False, action = 'store_true', help = 'Rescale the value of the data dividing by the value stored in rescale_factor. This is used in case the data where stored in uint16 (or similar format) and you want to rescale them to the 0-1 range. Default is False')
parser.add_argument('--use_normalization'   , default = True , action = 'store_true', help = 'Use normalization in the dataset. Default is True.')
parser.add_argument('--load_data_in_memory' , default = False, action = 'store_true', help = 'Load the data in memory. Default is False.')
# Boolean negate
parser.add_argument('--no-apply_rescale'      , dest = 'apply_rescale'      , action = 'store_false')
parser.add_argument('--no-use_normalization'  , dest = 'use_normalization'  , action = 'store_false')
parser.add_argument('--no-load_data_in_memory', dest = 'load_data_in_memory', action = 'store_false')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Load the dataset config
dataset_config = {}

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
if sum_percentages == 1.0 or sum_percentages == 0.9999999999999999 :
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the updated dataset config
with open(args.path_dataset_config, 'w') as f:
    toml.dump(dataset_config, f)
