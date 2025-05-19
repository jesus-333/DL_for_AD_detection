"""
As the name suggests this script will extract the middle slices from each scan and save them as independent samples.
This script is created to work with the output of convert/all_subjects_ADNI_ONLY_2D_MRI_V5_unpack.py
The data area saved as torch tensors (1 sample --> 1 tensor, conversion_type = 0), as a single tensor (all samples --> 1 tensor, conversion_type = 1) or both (conversion_type = 2)

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torchvision

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_data = "./data/ADNI_dicom_converted_axial/"
path_dataset_info = "./data/ADNI_dicom_converted_axial/data_info.json"

path_to_save = './data/ADNI_axial_middle_slice/'

conversion_type = 2

apply_resize = True
resize_size = 176

create_png_backup = True # If True for each slice saved it creates also a copy in png

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data and create preprocess function

# Get files and samples informations
list_of_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'npy')
with open(path_dataset_info) as f: data_info = json.load(f)

# (OPTIONAL) Function to resize data
if apply_resize :
    preprocess_function = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize_size, resize_size))
    ])
else :
    preprocess_function = None

# Create folder to save data
os.makedirs(path_to_save, exist_ok = True)
if create_png_backup : os.makedirs(path_to_save + 'png/', exist_ok = True)

# Variable to save slice informations
columns = ['file_name', 'subj_id', 'labels_int', 'labels_str', 'original_dataset']
slice_info_csv  = pd.DataFrame(columns = columns)
slice_info_dict = dict()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Extract and save middle slices

# Variables used to compute mean and std
channel_sum = 0
channel_squared_sum = 0
total_pixels = 0

# Variable for the single tensor dataset
dataset_tensor = []

n_element_to_print = len(list_of_files) // 50

for i in range(len(list_of_files)) :
    if i % n_element_to_print == 0 : print(f"Processing file {i}/{len(list_of_files)}\t({round(i / len(list_of_files) * 100, 2)}%)")

    # Get file and filename
    file_path = list_of_files[i]
    file_name = file_path.split('/')[-1]

    # Get sample info
    sample_info = data_info[file_name]
    z_matrix = sample_info['z_matrix']

    # Get sample info
    label_str = data_info[file_name]['label_str']
    label_int = data_info[file_name]['label_int']
    subj_id = data_info[file_name]['subj_id']
    original_dataset = data_info[file_name]['original_dataset']
    
    # Skip the sample with label Patient
    if label_str == 'Patient' : continue
    
    # Load sample and rescale in 0-1 range
    sample = torch.from_numpy(np.load(file_path))
    sample = sample / 4095

    # Get indices for the middle slides
    if z_matrix % 2 == 0 : # Even number of slices
        indices = [int(z_matrix / 2), int(z_matrix / 2 + 1)]
    else : # Odd numbero of slices
        indices = [z_matrix // 2 - 1, z_matrix // 2, z_matrix // 2 + 1]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Iterate through the middle slices
    for idx in indices :
        # Temporary variable to save slice info
        tmp_info_csv = []
        tmp_info_dict = dict()

        # Get slice
        slice = sample[idx].unsqueeze(0)

        # (OPTIONAL) Resize
        if apply_resize : slice = preprocess_function(slice)

        # Compute variable for mean/std evaluation
        channel_sum += torch.sum(slice)
        channel_squared_sum += torch.sum(slice ** 2)
        total_pixels += slice.shape[1] * slice.shape[2]

        # Save single slice
        if conversion_type == 0 or conversion_type == 2 :
            file_name_save = f'{i}_{idx}_{subj_id}_{label_str}.pth'
            file_path_save = f'{path_to_save}{file_name_save}'
            torch.save(slice, file_path_save)
        
        # Add slice to the dataset tensor
        if conversion_type == 1 or conversion_type == 2 :
            dataset_tensor.append(slice.unsqueeze(0))

        # Add info for the slice (csv)
        tmp_info_csv.append(file_name_save)
        tmp_info_csv.append(subj_id)
        tmp_info_csv.append(label_int)
        tmp_info_csv.append(label_str)
        tmp_info_csv.append(original_dataset)
        tmp_info_csv = np.asarray(tmp_info_csv)
        slice_info_csv = pd.concat([slice_info_csv, pd.DataFrame(tmp_info_csv.reshape(1, -1), columns = columns)], ignore_index = True)

        # Add info for the slice (dict)
        tmp_info_dict['subj_id'] = subj_id
        tmp_info_dict['label_int'] = label_int
        tmp_info_dict['label_str'] = label_str
        tmp_info_dict['original_dataset'] = original_dataset
        slice_info_dict[file_name_save] = tmp_info_dict

        # (OPTIONAL) Save png backup
        if create_png_backup :
            file_name_save = f'{i}_{idx}_{subj_id}_{label_str}.png'
            file_path_save = f'{path_to_save}png/{file_name_save}'
            matplotlib.image.imsave(file_path_save, slice.squeeze().numpy(), cmap = 'gray')

    # End cycle for slices
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# End cycle for files
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Save dataset tensor
if conversion_type == 1 or conversion_type == 2 :
    dataset_tensor = torch.cat(dataset_tensor)
    dataset_tensor_path = f'{path_to_save}dataset_tensor'
    if apply_resize : dataset_tensor_path += f'___{resize_size}_resize'
    torch.save(dataset_tensor, f"{dataset_tensor_path}.pt")

# Save slice info (csv)
slice_info_csv.to_csv(f'{path_to_save}data_info.csv', index = False)

# Save slice info (dict)
with open(f'{path_to_save}data_info.json', 'w') as f : json.dump(slice_info_dict, f, indent = 3)

# Compute mean and std and save them
mean = channel_sum / total_pixels
torch.save(mean, f'{path_to_save}dataset_mean.pt')

std = torch.sqrt((channel_squared_sum / total_pixels) - mean ** 2)
torch.save(std, f'{path_to_save}dataset_std.pt')

# Check mean and std against the dataset tensor
if conversion_type == 1 or conversion_type == 2 :
    mean_dataset_tensor = torch.mean(dataset_tensor, dim = (0, 2, 3))
    std_dataset_tensor = torch.std(dataset_tensor, dim = (0, 2, 3))
    print(f"Mean dataset tensor: {mean_dataset_tensor}")
    print(f"Std dataset tensor: {std_dataset_tensor}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
