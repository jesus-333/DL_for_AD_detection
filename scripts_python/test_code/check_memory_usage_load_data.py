"""
Track the memory usage when data are loaded.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse

# Create parser
parser = argparse.ArgumentParser(description = 'Update the demnet model configuration file with new parameters.')
parser.add_argument('--path_src'              , type = str , default = None, help = 'Path of the src folder (i.e. all the code related to model, dataset and training.)')
parser.add_argument('--path_data'             , type = str , default = None, help = 'Path to the folder with the data. If not provided, it will use the value defined in this script.')
parser.add_argument('--name_tensor_file'      , type = str , default = None, help = 'Name of the tensor file with the dataset. If not provided, it will use the default value defined in this script.')
parser.add_argument('--n_repetitions'         , type = int , default = None, help = 'Number of repetitions to measure the memory usage. Default is 5.')
parser.add_argument('--device_list'           , nargs = '+', default = []  , help = 'List of devices to use for the memory usage check. If empty, only cpu will be used.')

args = parser.parse_args()

# This specific import was added to allow the execution of the script with the "python" command from any folder you like.
# If the argument path_src is not provided, the script assume you will run it from the root folder of the repository.
import sys
if args.path_src is not None : sys.path.append(args.path_src)
else : sys.path.append('./')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import pandas as pd
import torch

from src.dataset import dataset, support_dataset
from src.training import support_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

percentage_train = 0.8
percentage_validation = 0.1
percentage_test = 0.1

percentage_split_list = [percentage_train, percentage_validation, percentage_test]

n_repetitions_script = 5
n_repetitions = n_repetitions_script if args.n_repetitions is None else args.n_repetitions

dataset_name = 'ADNI_axial_3D_z_48_size_176_int'
dataset_tensor_file_name = 'dataset_tensor___176_resize___int.pt' if args.name_tensor_file is None else args.name_tensor_file
path_to_data = f'./data/{dataset_name}/' if args.path_data is None else args.path_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

device_list_script = ['cpu']

device_list = device_list_script if len(args.device_list) == 0 else args.device_list

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

def load_data_train(use_mmap : bool, device : str) :
    data = load_data(use_mmap, device)
    data_validation  = data[idx_test]

    return data_train

def load_data_validation(use_mmap : bool, device : str) :
    data = load_data(use_mmap, device)
    data_validation  = data[idx_test]

    return data_validation

print("MEMORY USAGE WHEN DATA ARE LOADED. TRAIN VS VALIDATION")
for device in device_list :
    print(f"\nDevice: {device}")
    for use_mmap in [True, False]:
        print(f"\tUsing mmap: {use_mmap}")
        
        # Measure memory usage
        tmp_list_train = []
        tmp_list_validation = []
        for _ in range(n_repetitions):
            peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data_train, use_mmap = use_mmap, device = device)
            tmp_list_train.append(peak_memory_used)

            peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data_validation, use_mmap = use_mmap, device = device)
            tmp_list_validation.append(peak_memory_used)

        mean_memory_used = np.mean(tmp_list_train)
        std_memory_used = np.std(tmp_list_train)
        print(f"\tPeak memory used (TRAIN): {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")

        mean_memory_used = np.mean(tmp_list_validation)
        std_memory_used = np.std(tmp_list_validation)
        print(f"\tPeak memory used (VALIDATION): {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")
    print(" - - - - - - - - -")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# def load_data_and_create_dataset_V1(use_mmap : bool, device : str, use_clone : bool) :
#     data = load_data(use_mmap, device)
#
#     # Split data in train/validation/test
#     if use_clone :
#         MRI_train_dataset      = dataset.MRI_dataset(data[idx_train].clone()     , labels[idx_train]     , preprocess_functions = None, print_var = False)
#         MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation].clone(), labels[idx_validation], preprocess_functions = None, print_var = False)
#     else :
#         MRI_train_dataset      = dataset.MRI_dataset(data[idx_train]     , labels[idx_train]     , preprocess_functions = None, print_var = False)
#         MRI_validation_dataset = dataset.MRI_dataset(data[idx_validation], labels[idx_validation], preprocess_functions = None, print_var = False)
#
#     # Delete original data tensor to free memory
#     del data
#
#     return MRI_train_dataset, MRI_validation_dataset
#
# def load_data_and_create_dataset_V2(use_mmap : bool, device : str, use_clone : bool) :
#     data = load_data(use_mmap, device)
#
#     data_train = data[idx_train]
#     data_validation = data[idx_validation]
#     if use_clone :
#         data_train = data_train.clone()
#         data_validation = data_validation.clone()
#
#     del data
#
#     # Split data in train/validation/test
#     MRI_train_dataset      = dataset.MRI_dataset(data_train     , labels[idx_train]     , preprocess_functions = None, print_var = False)
#     MRI_validation_dataset = dataset.MRI_dataset(data_validation, labels[idx_validation], preprocess_functions = None, print_var = False)
#
#     # Delete original data tensor to free memory
#     # del data
#
#     return MRI_train_dataset, MRI_validation_dataset
#
# print("MEMORY USAGE WHEN DATA ARE LOADED AND DATASET IS CREATED (V1)")
# for device in device_list :
#     print(f"\nDevice: {device}")
#     for use_mmap in [True, False]:
#         print(f"\tUsing mmap: {use_mmap}")
#         
#         for use_clone in [True, False]:
#             print(f"\t\tUsing clone: {use_clone}")
#             
#             # Measure memory usage
#             tmp_list = []
#             for _ in range(n_repetitions):
#                 peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data_and_create_dataset_V1, use_mmap = use_mmap, device = device, use_clone = use_clone)
#                 tmp_list.append(peak_memory_used)
#
#             mean_memory_used = np.mean(tmp_list)
#             std_memory_used = np.std(tmp_list)
#             print(f"\t\tPeak memory used: {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")
#
#     print(" - - - - - - - - -")
#
#
# print("MEMORY USAGE WHEN DATA ARE LOADED AND DATASET IS CREATED (V2)")
# for device in device_list :
#     print(f"\nDevice: {device}")
#     for use_mmap in [True, False]:
#         print(f"\tUsing mmap: {use_mmap}")
#         
#         for use_clone in [True, False]:
#             print(f"\t\tUsing clone: {use_clone}")
#             
#             # Measure memory usage
#             tmp_list = []
#             for _ in range(n_repetitions):
#                 peak_memory_used = support_training.cpu_memory_usage_in_gb(load_data_and_create_dataset_V2, use_mmap = use_mmap, device = device, use_clone = use_clone)
#                 tmp_list.append(peak_memory_used)
#
#             mean_memory_used = np.mean(tmp_list)
#             std_memory_used = np.std(tmp_list)
#             print(f"\t\tPeak memory used: {mean_memory_used:.2f}±{std_memory_used:.2f} GB\n")
#
#     print(" - - - - - - - - -")
