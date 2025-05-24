"""
Given a list of 3D tensor, with the same shape (i.e. all tensor in the list must have the same shape D x H x W), this script merge all tensor of the list in a single tensor.
This script is created to work with the output obtained with convert/all_subjects_ADNI_2D_MRI_V4_4_torch.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import pandas as pd
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

list_of_paths_to_tensors = [
    './data/ADNI_axial_PD_z_35_slice_5_png_V4_3/dataset_tensor___176_resize___pixel_rescaling.pt',
    './data/ADNI_axial_PD_z_44_slice_4_png_V4_3/dataset_tensor___176 resize___pixel_rescaling___35_z_interpolated.pt',
    # './data/ADNI_axial_PD_z_48_slice_3_png_V4_3/dataset_tensor___176_resize___pixel_rescaling___35_z_interpolated.pt'
]

list_of_path_to_labels_files = [
    './data/ADNI_axial_PD_z_35_slice_5_png_V4_3/info_dataframe.csv',
    './data/ADNI_axial_PD_z_44_slice_4_png_V4_3/info_dataframe.csv',
    # './data/ADNI_axial_PD_z_48_slice_3_png_V4_3/info_dataframe.csv',
]

path_to_save_new_tensor = './data/ADNI_axial_PD_z_35_all_merged/'

compute_mean_and_std = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read data

if list_of_path_to_labels_files is not None :
    if len(list_of_paths_to_tensors) != len(list_of_path_to_labels_files) :
        raise ValueError(f"The number of tensors and the number of labels files must be the same. Please check the input files. The number of tensors is {len(list_of_paths_to_tensors)} and the number of labels files is {len(list_of_path_to_labels_files)}.")

os.makedirs(path_to_save_new_tensor, exist_ok = True)

tensors_list = []
info_dataframe_list = []

channel_sum = 0
channel_squared_sum = 0
total_pixels = 0
for i in range(len(list_of_paths_to_tensors)) :
    print(f"Reading {list_of_paths_to_tensors[i]}")
    tensors_list.append(torch.load(list_of_paths_to_tensors[i]))

    if compute_mean_and_std :
        channel_sum += torch.sum(tensors_list[i].type(torch.float) / 4095, dim = (0, 2, 3))
        channel_squared_sum += torch.sum((tensors_list[i].type(torch.float) / 4095) ** 2, dim = (0, 2, 3))
        shape_dataset = tensors_list[i].shape
        total_pixels += shape_dataset[0] * shape_dataset[2] * shape_dataset[3]

    if list_of_path_to_labels_files is not None :
        info_dataframe_list.append(pd.read_csv(list_of_path_to_labels_files[i]))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

torch.save(torch.cat(tensors_list), f"{path_to_save_new_tensor}dataset_tensor___176_resize___pixel_rescaling.pt")

info_dataframe_merged = pd.concat(info_dataframe_list, axis = 0, ignore_index = True).drop('Unnamed: 0', axis = 1)
info_dataframe_merged.to_csv(f'{path_to_save_new_tensor}info_dataframe.csv')

if compute_mean_and_std :
    mean = channel_sum / total_pixels
    print(mean)
    torch.save(mean, f'{path_to_save_new_tensor}dataset_mean.pt')

    std = torch.sqrt((channel_squared_sum / total_pixels) - mean ** 2)
    print(std)
    torch.save(std, f'{path_to_save_new_tensor}dataset_std.pt')
