"""
Download a pretrained demnet model and test it.
The model was trained on the Kaggle dataset "Alzheimer MRI 4 classes dataset".
More info about the dataset can be found here : https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset

The model will be tested on ADNI data (more info here https://adni.loni.usc.edu/data-samples/adni-data/).

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import numpy as np

from src.dataset import support_dataset, dataset
from src.model import demnet
from src.training import metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

load_early_stop_weights = True
epoch_to_load = -1

path_to_data = './data/ADNI_MRI_2D_Axial_png/'

print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model and data

# Download the trained model and get config used during model training.
# Note that the config is a dictionary with the keys : training_config, dataset_config, model_config
model, all_config = demnet.get_model_pretrained_with_kaggle_dataset(7, load_early_stop_weights, epoch_to_load)
print("Model DOWNLOADED")

# Get the config for model, training and dataset
model_config   = all_config['model_config']
training_config   = all_config['training_config']
dataset_config = all_config['dataset_config']

# Get the preprocess function used for the data
input_size = model_config['input_size']
grey_scale_image = True if model_config['input_channels'] == 1 else False
preprocess_functions, dataset_mean, dataset_std = support_dataset.get_preprocess_function('kaggle', input_size, grey_scale_image)

# Get data
list_of_path_to_data = [path_to_data + 'AD/', path_to_data + 'CN/', path_to_data + 'MCI/']
file_path_list, label_list_int, label_list_str = support_dataset.get_ADNI_dataset(list_of_path_to_data, print_var = print_var)


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

def compute_accuracy(n_samples, file_path_list, label_list_int, preprocess_functions, grey_scale_image, batch_size) :
    # Random samples n element
    idx = np.random.choice(len(file_path_list), n_samples, replace = False)
    file_path_list = [file_path_list[i] for i in idx]
    label_list_int = [label_list_int[i] for i in idx]
    
    # Create dataset and dataloader
    MRI_ADNI_all_dataset = dataset.MRI_2D_dataset(file_path_list, label_list_int, load_data_in_memory = False, preprocess_functions = preprocess_functions, grey_scale_image = grey_scale_image)
    MRI_ADNI_all_dataloader = torch.utils.data.DataLoader(MRI_ADNI_all_dataset, batch_size = batch_size, shuffle = True)

    print(f"Number of samples : {n_samples}")

    # Set the model to training mode
    model.train()

    # Evaluate the model on the train, validation and test set
    accuracy = metrics.compute_metrics(model, MRI_ADNI_all_dataloader, device)['accuracy']

    # Print results (train mode)
    print(f"\tAccuracy on train mode : {round(accuracy * 100, 2)}%")

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model on the train, validation and test set
    accuracy = metrics.compute_metrics(model, MRI_ADNI_all_dataloader, device)['accuracy']

    # Print results (eval mode)
    print(f"\tAccuracy on eval mode  : {round(accuracy * 100, 2)}%")


# Compute accuracy for different number of samples
n_samples_list = [1000, 5000, 10000, 50000]
batch_size = 92
for n_samples in n_samples_list : compute_accuracy(n_samples, file_path_list, label_list_int, preprocess_functions, grey_scale_image, batch_size)

# Hello... I have a problem with Amazon Gaming. More precisely I don't manage to activate it.
# Every time I tried I received the messagge "Add your mobile number to enable Prime Gaming".
# The problem is that I already have add my phone number in the amazon settings and I also enable the two step authentication.
# How could I solve this problem?
