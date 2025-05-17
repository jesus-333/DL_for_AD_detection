"""
Interpolate along the z-axis of axial image to change the number of slice.
This script is created to work with the single 3D dataset tensor obtained with convert/all_subjects_ADNI_2D_MRI_V4_4_torch.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom as dicom
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

z_interpolatation = 35
name_dataset = 'ADNI_axial_PD_z_48_slice_3_png_V4_3'
name_file_with_dataset_tensor = 'dataset_tensor___176_resize___pixel_rescaling.pt'
name_interpolated_file = f'dataset_tensor___176_resize___pixel_rescaling___{35}_z_interpolated.pt'

path_data = f'./data/{name_dataset}/{name_file_with_dataset_tensor}'
path_to_save_interpolated_data = f'./data/{name_dataset}/{name_interpolated_file}'

interpolation_mode = 'bilinear'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Interpolation

# Get data
dataset_to_interpolate = torch.load(path_data).type(torch.float)
print("Dataset loaded")

# New shape to used for the interpolation
new_shape = (z_interpolatation, dataset_to_interpolate.shape[3])

# Interpolate the data
dataset_interpolated = torch.nn.functional.interpolate(dataset_to_interpolate.permute(0, 2, 1, 3), size = new_shape, mode = interpolation_mode)
print("Interpolation ok")

# Get the original shape and cast to uint16
dataset_interpolated = dataset_interpolated.permute(0, 2, 1, 3).type(torch.uint16)

# Save dataset interpolated
torch.save(dataset_interpolated, path_to_save_interpolated_data)
