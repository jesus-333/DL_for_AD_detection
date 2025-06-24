"""
With this script you can fine tune the vgg neural network to classify MRI and fMRI data for alzheimer detection
For the dataset we used the the Kaggle alzheimer 4 class dataset (https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset/data)

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import toml
import numpy as np
import torch

from src.dataset import dataset_png, support_dataset
from src.model import vgg_nets
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_config = './scripts_training/config/vgg_finetuning.toml'

path_files_Moderate_Demented    = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/ModerateDemented'
path_files_Mild_Demented        = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/MildDemented'
path_files_Very_Mild_Demented   = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/VeryMildDemented'
path_files_Non_Demented         = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/NonDemented'

# This values are precomputed with the script compute_avg_std_dataset.py (withoug using the CenterCrop and Resize)
dataset_mean = torch.tensor([0.2816, 0.2816, 0.2816])
dataset_std  = torch.tensor([0.3269, 0.3269, 0.3269])

# This values are precomputed with the script compute_avg_std_dataset.py (using the CenterCrop and Resize before computation)
dataset_mean = torch.tensor([0.4233, 0.4233, 0.4233])
dataset_std  = torch.tensor([0.3179, 0.3179, 0.3179])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load config
all_config = toml.load(path_config)

training_config = all_config['training_config']
model_config = all_config['model_config']
dataset_config = all_config['dataset_config']

# Since the pth of model has size of ~0.5GB I save the only the model at the end of the training and the one with the lowest validation loss
training_config['epoch_to_save_model'] = training_config['epochs'] + 2

# Note that toml file din't have (yet) the null type
if training_config['seed'] == -1 : training_config['seed'] = None

# Save in the settings dataset_mean and dataset_std
dataset_config['dataset_mean'] = dataset_mean
dataset_config['dataset_std'] = dataset_std

# Wand Setting
training_config['wandb_training'] = True
training_config['project_name'] = "vgg_finetuning_AD"
training_config['name_training_run'] = None
training_config['model_artifact_name'] = "vgg_finetuning_AD_kaggle"
training_config['debug'] = False

# Percentage used to split data in train/validation/test
percentage_split_list = [dataset_config['percentage_train'], dataset_config['percentage_validation'], dataset_config['percentage_test']]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data path

label_to_int = dict(
    NonDemented = 0,
    ModerateDemented = 1,
    MildDemented = 2,
    VeryMildDemented = 3
)
if dataset_config['merge_AD_class'] : label_to_int['MildDemented'] = label_to_int['VeryMildDemented'] = label_to_int['ModerateDemented'] = 1

# Get the paths of the files for each class
file_path_list_1 = support_dataset.get_all_files_from_path(path_files_Moderate_Demented)
file_path_list_2 = support_dataset.get_all_files_from_path(path_files_Mild_Demented)
file_path_list_3 = support_dataset.get_all_files_from_path(path_files_Very_Mild_Demented)
file_path_list_control = support_dataset.get_all_files_from_path(path_files_Non_Demented)
print("Paths of files readed")

# Merge all paths and convert to nmupy array
file_path_list = file_path_list_1 + file_path_list_2 + file_path_list_3 + file_path_list_control
file_path_list = np.asarray(file_path_list)

# Get labels
label_list_str = support_dataset.get_labels_from_path_list(file_path_list)

# Convert labels
label_list_int = [label_to_int[label] for label in label_list_str]
label_list_int = np.asarray(label_list_int)

# Print the number of samples for each class
print("Number of samples for each class :")
if dataset_config['merge_AD_class'] :
    print("Control  : {}".format(np.sum(np.asarray(label_list_int) == 0)))
    print("Demented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
else :
    print("NonDemented      : {}".format(np.sum(np.asarray(label_list_int) == 0)))
    print("ModerateDemented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
    print("MildDemented     : {}".format(np.sum(np.asarray(label_list_int) == 2)))
    print("VeryMildDemented : {}".format(np.sum(np.asarray(label_list_int) == 3)))
print("Total number of samples : {}\n".format(len(label_list_int)))

# Get idx to split data in train, validation and test set
# idx_list = support_dataset.get_idx_to_split_data_V2(len(file_path_list), percentage_split_list, training_config['seed'])
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
vgg_model, preprocess_functions = vgg_nets.get_vgg(model_config)

if dataset_mean is not None : preprocess_functions.transforms[2].mean = dataset_mean
if dataset_std is not None : preprocess_functions.transforms[2].std = dataset_std

# Set type of finetuning
vgg_model.finetuning_type(training_config['training_mode'])
vgg_model.check_freeze_layer()

# Create datasets
load_data_in_memory = dataset_config['load_data_in_memory']
MRI_train_dataset      = dataset_png.MRI_2D_dataset(train_file_path_list, label_train_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions)
MRI_validation_dataset = dataset_png.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions)
MRI_test_dataset       = dataset_png.MRI_2D_dataset(test_file_path_list, label_test_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions)
print("Datasets CREATED")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train model

# vgg_model, training_metrics = train_functions.wandb_train(all_config, vgg_model, MRI_train_dataset, MRI_validation_dataset) 

# Used to check validity of get_idx_to_split_data_V3
# for i in range(4) : print(np.sum(label_train_list_int == i) / len(label_train_list_int) * 100)
# print("")
# for i in range(4) : print(np.sum(label_test_list_int == i) / len(label_test_list_int) * 100)
# print("")
# for i in range(4) : print(np.sum(label_validation_list_int == i) / len(label_validation_list_int) * 100)
# print("")
# for i in range(4) : print(np.sum(label_list_int == i) / len(label_list_int) * 100)
