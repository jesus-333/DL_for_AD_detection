"""
Given the path of the folder with the dicom files convert in npy files, this script take all the different scan and create a single dataset.
The conversion from dicom to npy file was done through the script convert/all_subjects_ADNI_2D_MRI_V5_unpack.py
The data area saved as torch tensors (1 sample --> 1 tensor, conversion_type = 0), as a single tensor (all samples --> 1 tensor, conversion_type = 1) or both (conversion_type = 2)

The arguments can be passed through the command line or set inside the script. If used, the argument from the command line will overwrite the ones set inside the script.

The scripts compute also the mean and standard deviation of the dataset, which are saved in separate tensors files.
The mean and standard deviation are computed for the input range [0, 1], regardless of the save_as_integer argument.
Note that the argument save_as_integer serves only to reduce the size of the saved tensors, in case memory is an issue.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)

P.s. I check the correspondance between the order of sample inside the dataset_info.csv file and the order of the tensors inside the dataset_tensor.pt file.
     The order is the same, so the first sample in the csv file corresponds to the first tensor in the dataset_tensor.pt file.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (OPTIONAL) CLI Arguments and check

import argparse

parser = argparse.ArgumentParser(description = 'Create the datasete from the files obtained from convert/all_subjects_ADNI_2D_MRI_V5_unpack.py')
parser.add_argument('--path_data'         , type = str, default = None, help = 'Path to the folder with the npy files. If None it must be specified in the script. If not specified an error is raised.')
parser.add_argument('--path_to_save'      , type = str, default = None, help = 'Path to the folder where the dataset will be saved. If not specified it is set to ./data/')
parser.add_argument('--conversion_type'   , type = int, default = None, help = 'Conversion type: 0 = 1 3D tensor for each sample (i.e. 1 sample, 1 file), 1 = A list of 3D tensors for all the dataset (i.e. all the dataset is saved in a list of tensors), 2 = both.')
parser.add_argument('--input_size'        , type = int, default = None, help = 'Size of the input image. By default, only the image with the size H or W with the specified input size will be saved. If apply_resize is passed, all the iamges will be resized to this size. Note, that in the case of images with only one dimension equal to the input size, the other dimension will be resized to the input size as well.')
parser.add_argument('--z_matrix'          , type = int, default = None, help = 'Number of slices in the z direction. If apply_z_matrix_interpolation is passed, the z-axis of all images will be interpolated to this size.')
parser.add_argument('--interpolation_mode', type = str, default = None, help = 'Interpolation mode for the z-axis interpolation. Other options are: nearest, area, nearest-exact.')
parser.add_argument('--n_samples'         , type = int, default = None, help = 'Number of samples to select randomly from the dataset. If a negative number is passed, all the samples will be used.')
parser.add_argument('--seed'              , type = int, default = None, help = 'Seed for the random number generator.')
parser.add_argument('--n_split'           , type = int, default = None, help = 'Number of splits to create. Works only if conversion_type is 1 or 2. If None, no split is created. If a positive number is passed, the dataset will be split in n_split parts and saved in separate files. The files will be named dataset_tensor_part_0.pt, dataset_tensor_part_1.pt, etc. If 0 (or a negative number) is passed, the dataset will not be split and saved in a single file named dataset_tensor.pt.')

# Boolean arguments
parser.add_argument('--apply_resize'                , default = None, action = 'store_true', help = 'If passed it applies a resize to the data. Default is True.')
parser.add_argument('--apply_z_matrix_interpolation', default = None, action = 'store_true', help = 'If passed it applies a z-axis interpolation to the data. Default is False.')
parser.add_argument('--save_as_integer'             , default = None, action = 'store_true', help = 'If passed it saves the data as integer (uint16) in the [0, 4095] range. Otherwise the data will be save as float32 in the range [0, 1]. Note that 4095 (or 1 if you convert to float) is the maximum value of the pixel, but not necessarly all images reach this value.')

args = parser.parse_args()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pandas as pd
import torch
import torchvision

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Arguments (inside the scripts)

path_data = './data/ADNI_dicom_converted_axial/'

conversion_type = 1

input_size = 176
apply_resize = True

z_matrix = 48
apply_z_matrix_interpolation = False
interpolation_mode = 'bilinear'  # 'nearest', 'linear', 'area', 'nearest-exact'

n_samples = -1
n_split = 0
seed = 42

path_to_save = f'./data/ADNI_axial_3D_z_{z_matrix}_size_{input_size}/'

save_as_integer = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check arguments

# If the arguments are passed through the command line, they will overwrite the ones set inside the script
if args.path_data is not None       : path_data = args.path_data
if args.path_to_save is not None    : path_to_save = args.path_to_save
if args.conversion_type is not None : conversion_type = args.conversion_type
if args.input_size is not None      : input_size = args.input_size
if args.z_matrix is not None        : z_matrix = args.z_matrix
if args.n_samples is not None       : n_samples = args.n_samples
if args.seed is not None            : seed = args.seed
if args.apply_resize is not None    : apply_resize = args.apply_resize
if args.save_as_integer is not None : save_as_integer = args.save_as_integer
if args.apply_z_matrix_interpolation is not None :
    apply_z_matrix_interpolation = args.apply_z_matrix_interpolation
    if args.interpolation_mode is not None : interpolation_mode = args.interpolation_mode

# Check if some arguments are still None
if path_data is None:
    raise ValueError("The path_data argument must be specified either in the script or through the command line.")
if path_to_save is None:
    path_to_save = './data/'  # Default path if not specified
    print(f"Warning: path_to_save not specified. Using default path: {path_to_save}")
if conversion_type is None:
    conversion_type = 1  # Default conversion type if not specified
    print(f"Warning: conversion_type not specified. Using default value: {conversion_type} (1 = single tensor for all samples)")
if input_size is None:
    raise ValueError("The input_size argument must be specified either in the script or through the command line.")
if z_matrix is None:
    raise ValueError("The z_matrix argument must be specified either in the script or through the command line.")
if n_samples is None:
    n_samples = -1  # Default to -1 (use all samples) if not specified
    print(f"Warning: n_samples not specified. Using default value: {n_samples} (use all samples)")
if seed is None:
    seed = 42  # Default seed if not specified
    print(f"Warning: seed not specified. Using default value: {seed}")
if apply_resize is None:
    apply_resize = True  # Default to True if not specified
    print(f"Warning: apply_resize not specified. Using default value: {apply_resize} (True)")
if apply_z_matrix_interpolation is None:
    apply_z_matrix_interpolation = False  # Default to False if not specified
    print(f"Warning: apply_z_matrix_interpolation not specified. Using default value: {apply_z_matrix_interpolation} (False)")
elif apply_z_matrix_interpolation and interpolation_mode is None:
    interpolation_mode = 'bilinear'
    print("Warning: apply_z_matrix_interpolation is True but interpolation_mode is not specified. Using default value: bilinear.")

# Set the random seed for reproducibility
np.random.seed(seed)

# Create the path to save the dataset if it does not exist
os.makedirs(path_to_save, exist_ok = True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get the data

data_info = pd.read_csv(f"{path_data}data_info.csv")

# Filter by input_size if apply_resize is False
if not apply_resize: data_info = data_info[(data_info['H'] == input_size) | (data_info['W'] == input_size)]

# Filter by z_matrix if apply_z_matrix_interpolation is False
if not apply_z_matrix_interpolation: data_info = data_info[data_info['z_matrix'] == z_matrix]

# Randomly sample n_samples if n_samples > 0:
if n_samples > 0: data_info = data_info.sample(n = n_samples, random_state = seed, replace = False)

# Resize the images to the input size
preprocess_function = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size))
])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Variable used to store the tensors of the samples
dataset_tensor = []

# Variables used to compute the mean and standard deviation of the dataset
channel_sum = 0
channel_squared_sum = 0
total_pixels = 0

n_element_to_print = int(len(data_info) / 252)
if n_element_to_print == 0 : n_element_to_print = 1  # Ensure at least one print

idx_patient_to_remove = []

for i in range(len(data_info)):
    if i % n_element_to_print == 0 : print(f"Processing file {i}/{len(data_info)}\t({round(i / len(data_info) * 100, 2)}%)")

    # Get the sample info
    sample_info = data_info.iloc[i]

    if sample_info['labels_str'] == 'Patient' :
        idx_patient_to_remove.append(i)
        continue

    # Get the path
    npy_file_path = os.path.join(path_data, sample_info['file_name'])
    # Load the npy file
    sample_3D_tensor = torch.from_numpy(np.load(npy_file_path))

    # Resize if apply_resize is True
    if apply_resize:
        sample_3D_tensor = preprocess_function(sample_3D_tensor)

    # Interpolate the z-axis if apply_z_matrix_interpolation is True
    if apply_z_matrix_interpolation:
        # Get the new shape (note that the z-axis dimension is moved to the spatial axis)
        new_shape = (z_matrix, input_size)

        # Convert to float32 (interpolation does not work on uint16)
        sample_3D_tensor = sample_3D_tensor.float()
        
        # Interpolate the z-axis of the sample
        # The permute is done to move the z-axis to the spatial axis and allow the interpolation.
        # More info here https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        sample_3D_tensor = torch.nn.functional.interpolate(sample_3D_tensor.permute(1, 0, 2).unsqueeze(0), size = new_shape, mode = interpolation_mode).squeeze(0).permute(1, 0, 2)

    # Compute the mean and std
    channel_sum += torch.sum(sample_3D_tensor.type(torch.float) / 4095, dim = (1, 2))
    channel_squared_sum += torch.sum((sample_3D_tensor.type(torch.float) / 4095) ** 2, dim = (1, 2))
    shape_sample = sample_3D_tensor.shape
    total_pixels += shape_sample[0] * shape_sample[1] * shape_sample[2]

    # Type conversion
    if save_as_integer:
        # Convert to uint16 in the range [0, 4095]
        sample_3D_tensor = sample_3D_tensor.to(torch.int16)
    else:
        # Convert to float32 in the range [0, 1]
        sample_3D_tensor = sample_3D_tensor.float() / 4095.0

    # Save the tensor if conversion_type is 0 or 2
    if conversion_type == 0 or conversion_type == 2:
        # Save the tensor as a single file
        file_name = sample_info['file_name'].replace('.npy', '.pt')
        sample_tensor_path = os.path.join(path_to_save, file_name)
        torch.save(sample_3D_tensor, sample_tensor_path)

    # Append to the dataset tensor
    dataset_tensor.append(sample_3D_tensor)

    # END for loop

# Convert the list of tensors to a single tensor if conversion_type is 1 or 2
if conversion_type == 1 or conversion_type == 2:
    dataset_tensor = torch.stack(dataset_tensor)

    # Save the dataset tensor
    if n_split <= 0:
        dataset_tensor_path = os.path.join(path_to_save, 'dataset_tensor')
        if apply_resize : dataset_tensor_path += '___176_resize'
        if save_as_integer : dataset_tensor_path += '___int'
        dataset_tensor_path += '.pt'
        torch.save(dataset_tensor, dataset_tensor_path)
    else :
        # Split the dataset tensor into n_split parts and save them
        split_size = len(dataset_tensor) // n_split
        for j in range(n_split):
            # Get the start and end indices for the split
            start_idx = j * split_size
            end_idx = (j + 1) * split_size if j < n_split - 1 else len(dataset_tensor)

            # Extract the part of the tensor and save it
            part_tensor = dataset_tensor[start_idx:end_idx].clone()
            part_tensor_path = os.path.join(path_to_save, f'dataset_tensor_part_{j}')
            if apply_resize : part_tensor_path += '___176_resize'
            if save_as_integer : part_tensor_path += '___int'
            part_tensor_path += '.pt'
            torch.save(part_tensor, part_tensor_path)

# Save the info dataframe
data_info.drop(index = data_info.index[idx_patient_to_remove], axis = 1, inplace = True)
info_dataframe_path = os.path.join(path_to_save, 'dataset_info.csv')
data_info.to_csv(info_dataframe_path, index = False)

# Compute dataset mean and std, and save them
mean = channel_sum / total_pixels
torch.save(mean, f'{path_to_save}dataset_mean.pt')

std = torch.sqrt((channel_squared_sum / total_pixels) - mean ** 2)
torch.save(std, f'{path_to_save}dataset_std.pt')
