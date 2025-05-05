"""
With this script you can train the DEMNET model to classify MRI and fMRI data for alzheimer detection.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692

This version is used to training the model on THE converted ADNI Dataset.
I.e. I download the dataset (using the 2D filter), convert all the images in png and keep each recording of the subject as a sample of the dataset (see the script convert_all_subjects_ADNI_ONLY_2D_MRI_V4_2.py).

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import toml
import numpy as np
import torch
import json
from torchvision import transforms

from src.dataset import dataset, support_dataset, support_dataset_ADNI
from src.model import demnet
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_config_train_and_dataset = './scripts/training/config/demnet_training_and_dataset.toml'
path_config_model             = './scripts/training/config/demnet_model.toml'

dataset_name = 'ADNI_axial_PD_z_44_slice_4'
path_to_data = f'./data/{dataset_name}_png_V4_2/'
z_matrix = 44 # Number of slice per sample

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load train and dataset config
train_and_dataset_config = toml.load(path_config_train_and_dataset)
train_config = train_and_dataset_config['train_config']
dataset_config = train_and_dataset_config['dataset_config']

# Load model config
model_config = toml.load(path_config_model)
model_config['input_channels'] = z_matrix

# Create single dictionary with all the config
all_config = dict(
    train_config = train_config,
    dataset_config = dataset_config,
    model_config = model_config
)

if 'path_to_data' in dataset_config : path_to_data = dataset_config['path_to_data']

# train_config['epoch_to_save_model'] = train_config['epochs'] + 2
# Note that toml file din't have (yet) the null type
if train_config['seed'] == -1 : train_config['seed'] = np.random.randint(0, 1e9)

if model_config['input_size'] == 224 :
    raise ValueError("Value not actually computed")
elif model_config['input_size'] == 176 :
    # This values are precomputed with the script compute_avg_std_dataset.py (using the Resize(176)  before computation)
    mean_to_use = 0.14556307
    std_to_use  = 0.17802857 
    dataset_mean = torch.tensor([mean_to_use, mean_to_use, mean_to_use]) if not dataset_config['grey_scale_image'] else torch.tensor([mean_to_use])
    dataset_std  = torch.tensor([std_to_use, std_to_use, std_to_use]) if not dataset_config['grey_scale_image'] else torch.tensor([std_to_use])

    tmp_list = [transforms.Resize((model_config['input_size'], model_config['input_size']))]
    if dataset_config['use_normalization'] : tmp_list.append(transforms.Normalize(mean = dataset_mean, std = dataset_std))

    preprocess_functions  = transforms.Compose(tmp_list)
else :
    raise ValueError("dataset_config['input_size'] value not valid")

# Save in the settings dataset_mean and dataset_std
if dataset_config['use_normalization'] :
    dataset_config['dataset_mean'] = dataset_mean
    dataset_config['dataset_std'] = dataset_std

# Wand Setting
train_config['wandb_training'] = True
train_config['project_name'] = "demnet_training_ADNI"
train_config['name_training_run'] = None
train_config['model_artifact_name'] = "demnet_training_ADNI"

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files divided per folder
folders_paths_dict = support_dataset.get_all_files_from_path_divided_per_folder(path_to_data, filetype_filter = 'png')

# Get depth map order for each folder
depth_map_order_dict = support_dataset_ADNI.get_depth_map_order_all_dataset(folders_paths_dict)

# Get labels
with open(f'./data/ADNI_Labels/{dataset_name}_int.json') as fp: subj_to_label_int = json.load(fp)
with open(f'./data/ADNI_Labels/{dataset_name}_str.json') as fp: subj_to_label_str = json.load(fp)
label_list_int = support_dataset_ADNI.get_labels_from_path_dict_V4_2(folders_paths_dict, subj_to_label_int)
label_list_str = support_dataset_ADNI.get_labels_from_path_dict_V4_2(folders_paths_dict, subj_to_label_str)

idx_list = support_dataset.get_idx_to_split_data_V3(label_list_int, percentage_split_list, train_config['seed'])
idx_train, idx_validation, idx_test = idx_list

raise ValueError("STOP")

# Save indices in the config
train_config['idx_train']      = idx_train
train_config['idx_test']       = idx_test
train_config['idx_validation'] = idx_validation

# Split the data
train_file_path_list,      label_train_list_int      = file_path_list[idx_train],      label_list_int[idx_train]
validation_file_path_list, label_validation_list_int = file_path_list[idx_validation], label_list_int[idx_validation]
test_file_path_list,       label_test_list_int       = file_path_list[idx_test],       label_list_int[idx_test]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Select training device 

if torch.cuda.is_available() :
    device = torch.device("cuda")
    print("\nCUDA backend in use")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nmps backend (apple metal) in use")
else:
    device = torch.device("cpu")
    print("\nNo backend in use. Device set to cpu")
train_config['device'] = device

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load model
model_config['num_classes'] = len(set(label_list_int))
model = demnet.demnet(model_config)

# Create datasets
load_data_in_memory = dataset_config['load_data_in_memory']
MRI_train_dataset      = dataset.MRI_2D_dataset(train_file_path_list, label_train_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
MRI_validation_dataset = dataset.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
MRI_test_dataset       = dataset.MRI_2D_dataset(test_file_path_list, label_test_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
print("\nDatasets CREATED")
print(f"\tSamples used = {dataset_config['n_samples']}")
print(f"\tTrain samples      = {len(MRI_train_dataset)}")
print(f"\tValidation samples = {len(MRI_validation_dataset)}")
print(f"\tTest samples       = {len(MRI_test_dataset)}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Train model

model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset) 
