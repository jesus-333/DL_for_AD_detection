"""
Check if the training functions are working correctly
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import toml
import torch

from src.dataset import dataset, support_dataset
from src.model import vgg_nets
from src.training import train_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_files_Moderate_Demented    = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/ModerateDemented'
path_files_Mild_Demented        = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/MildDemented'
path_files_Very_Mild_Demented   = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/VeryMildDemented'
path_files_Non_Demented         = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/NonDemented'

merge_all_demented_class = True
load_data_in_memory = True
batch_size = 32
epochs = 2

percentage_train = 0.5
percentage_validation = 0.2
percentage_test = 0.3
percentage_split_list = [percentage_train, percentage_validation, percentage_test]

seed = None

path_train_config = './scripts_training/config/vgg_finetuning.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Datasets creation

# Dictionary used to convert textural labels to integers
label_to_int = dict(
    NonDemented = 0,
    ModerateDemented = 1,
    MildDemented = 2,
    VeryMildDemented = 3
)
if merge_all_demented_class : label_to_int['MildDemented'] = label_to_int['VeryMildDemented'] = label_to_int['ModerateDemented'] = 1

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
if merge_all_demented_class :
    print("Control  : {}".format(np.sum(np.asarray(label_list_int) == 0)))
    print("Demented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
else :
    print("NonDemented      : {}".format(np.sum(np.asarray(label_list_int) == 0)))
    print("ModerateDemented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
    print("MildDemented     : {}".format(np.sum(np.asarray(label_list_int) == 2)))
    print("VeryMildDemented : {}".format(np.sum(np.asarray(label_list_int) == 3)))
print("Total number of samples : {}\n".format(len(label_list_int)))

# Get idx to split data in train, validation and test set
idx_list = support_dataset.get_idx_to_split_data_V2(len(file_path_list), percentage_split_list, seed)
idx_train, idx_validation, idx_test = idx_list

# Check if the idx are correct
print("Check if the idx are correct")
print(f"Total number of samples                   : {len(file_path_list)}")
print(f"Number of samples for each split          : {[len(idx) for idx in idx_list]}")
print(f"Number of samples if I merge all idx list : {np.sum([len(idx) for idx in idx_list])}\n")

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
# Create model and dataset 

# Get config
config = toml.load(path_train_config)

# Setup training config to test the training function
config['train_config']['batch_size'] = batch_size
config['train_config']['epochs'] = epochs
config['train_config']['epoch_to_save_model'] = config['train_config']['epochs'] + 2
config['train_config']['measure_metrics_during_training'] = True
config['train_config']['device'] = device

# Get model
vgg_model, preprocess_functions = vgg_nets.get_vgg(config['model_config'])
print("Model CREATED")

# Create datasets
MRI_train_dataset      = dataset.MRI_2D_dataset(train_file_path_list, label_train_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions)
MRI_validation_dataset = dataset.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions)
MRI_test_dataset       = dataset.MRI_2D_dataset(test_file_path_list, label_test_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions)
print("Datasets CREATED")

# Check training 
model = train_functions.train(config['train_config'], vgg_model, MRI_train_dataset, MRI_validation_dataset) 

