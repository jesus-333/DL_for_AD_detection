"""
Download a pretrained demnet model and test it. 
The model was trained on the Kaggle dataset "Alzheimer MRI 4 classes dataset". 
More info about the dataset can be found here : https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset

The model is tested on the train, validation and test set (splitted as in the training phase).

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch

from src.dataset import support_dataset, dataset
from src.model import demnet
from src.training import metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

version = 8
load_early_stop_weights = True
epoch_to_load = -1

path_to_download_dataset = './data/'

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get model and data

# Download the trained model and get config used during model training.
# Note that the config is a dictionary with the keys : training_config, dataset_config, model_config
model, all_config = demnet.get_model_pretrained_with_kaggle_dataset(version, load_early_stop_weights, epoch_to_load)
print("Model DOWNLOADED")

# Get the config for model, training and dataset
model_config   = all_config['model_config']
training_config   = all_config['training_config']
dataset_config = all_config['dataset_config']

# Download the Kaggle dataset "Alzheimer MRI 4 classes dataset"
merge_AD_class = dataset_config['merge_AD_class']
file_path_list, label_list_int, label_list_str = support_dataset.download_and_get_kaggle_AD_dataset(path_to_download_dataset, merge_AD_class, print_var)
print("Data DOWNLOADED")

# Get the preprocess function used for the data
input_size = model_config['input_size']
grey_scale_image = True if model_config['input_channels'] == 1 else False
preprocess_functions, dataset_mean, dataset_std = support_dataset.get_preprocess_function('kaggle', input_size, grey_scale_image)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Create the dataset

# Get the indices used to split the data before training.
# Note that you can do this only because I add the indices in the config file before training.
# In this way the indices are uploaded with tht model in wandb.
idx_train      = training_config['idx_train']
idx_test       = training_config['idx_test']
idx_validation = training_config['idx_validation']

# Split the data
train_file_path_list,      label_train_list_int      = file_path_list[idx_train],      label_list_int[idx_train]
validation_file_path_list, label_validation_list_int = file_path_list[idx_validation], label_list_int[idx_validation]
test_file_path_list,       label_test_list_int       = file_path_list[idx_test],       label_list_int[idx_test]

load_data_in_memory = dataset_config['load_data_in_memory']
MRI_train_dataset      = dataset.MRI_2D_dataset(train_file_path_list, label_train_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
MRI_validation_dataset = dataset.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
MRI_test_dataset       = dataset.MRI_2D_dataset(test_file_path_list, label_test_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image'])
print("Datasets CREATED")

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
# Compute the accuracy of the model on the train, validation and test set

MRI_train_dataloader = torch.utils.data.DataLoader(MRI_train_dataset, batch_size = training_config['batch_size'], shuffle = True)
MRI_validation_dataloader = torch.utils.data.DataLoader(MRI_validation_dataset, batch_size = training_config['batch_size'], shuffle = False)
MRI_test_dataloader = torch.utils.data.DataLoader(MRI_test_dataset, batch_size = training_config['batch_size'], shuffle = False)

# Evaluate the model on the train, validation and test set
accuracy_train = metrics.compute_metrics(model, MRI_train_dataloader, device)['accuracy']
accuracy_validation = metrics.compute_metrics(model, MRI_validation_dataloader, device)['accuracy']
accuracy_test = metrics.compute_metrics(model, MRI_test_dataloader, device)['accuracy']

# Set the model to training mode
model.train()

# Print results (train mode)
print("Accuracy with model in training mode")
print(f"\tAccuracy on train set      : {round(accuracy_train * 100, 2)}%")
print(f"\tAccuracy on validation set : {round(accuracy_validation * 100, 2)}%")
print(f"\tAccuracy on test set       : {round(accuracy_test * 100, 2)}%")

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the train, validation and test set
accuracy_train = metrics.compute_metrics(model, MRI_train_dataloader, device)['accuracy']
accuracy_validation = metrics.compute_metrics(model, MRI_validation_dataloader, device)['accuracy']
accuracy_test = metrics.compute_metrics(model, MRI_test_dataloader, device)['accuracy']

# Print results (eval mode)
print("Accuracy with model in evaluation mode")
print(f"\tAccuracy on train set      : {round(accuracy_train * 100, 2)}%")
print(f"\tAccuracy on validation set : {round(accuracy_validation * 100, 2)}%")
print(f"\tAccuracy on test set       : {round(accuracy_test * 100, 2)}%")

