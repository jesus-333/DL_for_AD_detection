"""
Track the memory usage when data are loaded.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd
import toml
import torch
import torchvision

from src.dataset import dataset, support_dataset
from src.model import demnet
from src.training import support_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

dataset_name = 'ADNI_axial_3D_z_48_size_176_int'
dataset_tensor_file_name = 'dataset_tensor___176_resize___int.pt'
path_to_data = f'./data/{dataset_name}/'

percentage_train = 0.8
percentage_validation = 0.1
percentage_test = 0.1

percentage_split_list = [percentage_train, percentage_validation, percentage_test]

n_repetitions = 5

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

device_list = ['cpu']
if torch.cuda.is_available() : device_list.append('cuda')
if torch.backends.mps.is_available() : device_list.append('mps')

seed = np.random.randint(0, 1e9)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check memory usage only when data are loaded

def load_data(use_mmap : bool, device : str) :
    if device == 'cuda' : support_training.gpu_start_memory_tracking()
    data = torch.load(f'{path_to_data}{dataset_tensor_file_name}', mmap = use_mmap, map_location = device)
    if device == 'cuda' : support_training.gpu_stop_memory_tracking()

    return data

print("MEMORY USAGE WHEN DATA ARE LOADED")
for device in device_list :
    print(f"\nDevice: {device}")
    for use_mmap in [True, False]:
        print(f"\tUsing mmap: {use_mmap}")
        
        # Measure memory usage
        tmp_list = []
        for _ in range(n_repetitions):
            peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data, use_mmap = use_mmap, device = device)
            tmp_list.append(peak_memory_used)

        mean_memory_used = np.mean(tmp_list)
        std_memory_used = np.std(tmp_list)
        print(f"\tPeak memory used: {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")
    print(" - - - - - - - - -")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get labels
dataset_info = pd.read_csv(f'{path_to_data}dataset_info.csv')
labels = dataset_info['labels_int'].to_numpy()
labels_str = dataset_info['labels_str'].to_numpy()

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels, percentage_split_list, seed)
idx_train, idx_validation, idx_test = idx_list

# Create random indices to train/validation/test split
# P.s. this function has the side effect to sort the samples according to labels (so the first you will have all the samples with label 0, then all the samples with label 1 and so on)
idx_list = support_dataset.get_idx_to_split_data_V3(labels, percentage_split_list, seed)
idx_train, idx_validation, idx_test = idx_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# def load_data_and_normalize_data(use_mmap : bool, device : str) :
#     data = load_data(use_mmap, device)
#
#     # Normalize data
#     data = data / 4095
#
#     return None
#
# print("MEMORY USAGE WHEN DATA ARE LOADED AND NORMALIZED")
# for device in device_list :
#     print(f"\nDevice: {device}")
#     for use_mmap in [True, False]:
#         print(f"\tUsing mmap: {use_mmap}")
#         
#         # Measure memory usage
#         tmp_list = []
#         for _ in range(n_repetitions):
#             peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data_and_normalize_data, use_mmap = use_mmap, device = device)
#             tmp_list.append(peak_memory_used)
#
#         mean_memory_used = np.mean(tmp_list)
#         std_memory_used = np.std(tmp_list)
#         print(f"\tPeak memory used: {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")
#     print(" - - - - - - - - -")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def load_data_and_create_dataset(use_mmap : bool, device : str, use_clone : bool) :
    data = load_data(use_mmap, device)

    # Split data in train/validation/test
    if use_clone :
        MRI_train_dataset      = dataset.MRI_dataset(data[idx_train].clone()      , labels[idx_train]     , preprocess_functions = None, print_var = False)
        MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation].clone() , labels[idx_validation], preprocess_functions = None, print_var = False)
    else :
        MRI_train_dataset      = dataset.MRI_dataset(data[idx_train]      , labels[idx_train]     , preprocess_functions = None, print_var = False)
        MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation] , labels[idx_validation], preprocess_functions = None, print_var = False)

    # Delete original data tensor to free memory
    # del data

    return MRI_train_dataset, MRI_validation_dataset

print("MEMORY USAGE WHEN DATA ARE LOADED AND DATASET IS CREATED")
for device in device_list :
    print(f"\nDevice: {device}")
    for use_mmap in [True, False]:
        print(f"\tUsing mmap: {use_mmap}")
        
        for use_clone in [True, False]:
            print(f"\tUsing clone: {use_clone}")
            
            # Measure memory usage
            tmp_list = []
            for _ in range(n_repetitions):
                peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data_and_create_dataset, use_mmap = use_mmap, device = device, use_clone = use_clone)
                tmp_list.append(peak_memory_used)

            mean_memory_used = np.mean(tmp_list)
            std_memory_used = np.std(tmp_list)
            print(f"\tPeak memory used: {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")

    print(" - - - - - - - - -")
