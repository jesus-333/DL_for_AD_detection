"""
With this script you can train the DEMNET model to classify MRI and fMRI data for alzheimer detection.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692 
For the dataset we used the the Kaggle alzheimer 4 class dataset (https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset/data)

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import toml
import numpy as np
import torch
from torchvision import transforms

from src.dataset import dataset_png, support_dataset, support_dataset_kaggle
from src.model import demnet
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_config_train_and_dataset = './scripts/training/config/demnet_training_and_dataset.toml'
path_config_model             = './scripts/training/config/demnet_model.toml'

path_files_Moderate_Demented    = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/ModerateDemented'
path_files_Mild_Demented        = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/MildDemented'
path_files_Very_Mild_Demented   = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/VeryMildDemented'
path_files_Non_Demented         = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/NonDemented'

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load train and dataset config
train_and_dataset_config = toml.load(path_config_train_and_dataset)
training_config = train_and_dataset_config['training_config']
dataset_config = train_and_dataset_config['dataset_config']

# Load model config
model_config = toml.load(path_config_model)
model_config['input_channels'] = 1 if dataset_config['grey_scale_image'] else 3

# Create single dictionary with all the config
all_config = dict(
    training_config = training_config,
    dataset_config = dataset_config,
    model_config = model_config
)

# training_config['epoch_to_save_model'] = training_config['epochs'] + 2

# Note that toml file din't have (yet) the null type
if training_config['seed'] == -1 : training_config['seed'] = np.random.randint(0, 1e8)

# This values are precomputed with the script compute_avg_std_dataset.py (withoug using the CenterCrop and Resize)
# dataset_mean = torch.tensor([0.2816, 0.2816, 0.2816])
# dataset_std  = torch.tensor([0.3269, 0.3269, 0.3269])

# This values are precomputed with the script compute_avg_std_dataset.py (using the Resize(256) and CenterCrop(224) before computation)
if model_config['input_size'] == 224 :
    dataset_mean = torch.tensor([0.4233, 0.4233, 0.4233]) if not dataset_config['grey_scale_image'] else torch.tensor([0.4233])
    dataset_std  = torch.tensor([0.3179, 0.3179, 0.3179]) if not dataset_config['grey_scale_image'] else torch.tensor([0.3179])

    tmp_list = [transforms.Resize((model_config['input_size'], model_config['input_size']))]
    if dataset_config['use_normalization'] : tmp_list.append(transforms.Normalize(mean = dataset_mean, std = dataset_std))

    preprocess_functions  = transforms.Compose(tmp_list)
elif model_config['input_size'] == 176 :
    # This values are precomputed with the script compute_avg_std_dataset.py (using the Resize(176)  before computation)
    dataset_mean = torch.tensor([0.2816, 0.2816, 0.2816]) if not dataset_config['grey_scale_image'] else torch.tensor([0.2816])
    dataset_std  = torch.tensor([0.3259, 0.3259, 0.3259]) if not dataset_config['grey_scale_image'] else torch.tensor([0.3259])

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
training_config['wandb_training'] = True
training_config['project_name'] = "demnet_training"
training_config['model_artifact_name'] = "demnet_training_AD_kaggle"
if 'name_training_run' in training_config :
    if training_config['name_training_run'] == "" : training_config['name_training_run'] = None
else :
    training_config['name_training_run'] = None

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data path
file_path_list, label_list_int, label_list_str = support_dataset_kaggle.get_data(path_files_Moderate_Demented, path_files_Mild_Demented, path_files_Very_Mild_Demented, path_files_Non_Demented, 
                                                                                 dataset_config['merge_AD_class'], print_var)

idx_list = support_dataset.get_idx_to_split_data_V3(label_list_int, percentage_split_list, training_config['seed'])
idx_train, idx_validation, idx_test = idx_list

# Save indices in the config
training_config['idx_train']      = idx_train
training_config['idx_test']       = idx_test
training_config['idx_validation'] = idx_validation

# Split the data
train_file_path_list,      label_train_list_int      = file_path_list[idx_train],      label_list_int[idx_train]
validation_file_path_list, label_validation_list_int = file_path_list[idx_validation], label_list_int[idx_validation]
test_file_path_list,       label_test_list_int       = file_path_list[idx_test],       label_list_int[idx_test]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Select training device 

if torch.cuda.is_available() :
    device = torch.device("cuda")
    print("CUDA backend in use")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("mps backend (apple metal) in use")
else:
    device = torch.device("cpu")
    print("No backend in use. Device set to cpu")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load model
model_config['num_classes'] = len(set(label_list_int))
model = demnet.demnet(model_config)

# Create datasets
load_data_in_memory = dataset_config['load_data_in_memory']
MRI_train_dataset      = dataset_png.MRI_2D_dataset(train_file_path_list, label_train_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
MRI_validation_dataset = dataset_png.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
MRI_test_dataset       = dataset_png.MRI_2D_dataset(test_file_path_list, label_test_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
print("Datasets CREATED")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model

model, training_metrics = train_functions.wandb_train(all_config, model, MRI_train_dataset, MRI_validation_dataset) 
