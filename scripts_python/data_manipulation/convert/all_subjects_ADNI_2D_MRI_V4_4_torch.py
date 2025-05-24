"""
It is similar to V4_3, i.e. conversion of dcm in file tensor, but instead of numpy it will use torch.
I keep it separate from v4_3 to avoid having scripts that are too long/complicated/with too many variable.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import numpy as np
import os
import pandas as pd
import pydicom as dicom
import torch
import torchvision

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_48_slice_3"
path_all_data = f"./data/{dataset_name}/"
path_to_save = f"./data/{dataset_name}_png_V4_3/"

conversion_type = 1 # 0 = 1 3D tensor for each sample (i.e. 1 sample, 1 file), 1 = A list of 3D tensors for all the dataset (i.e. all the dataset is saved in a list of tensors), 2 = both
resize_size = 176

apply_resize = True
apply_pixel_rescaling = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')
list_converted_folders = []
info_array = []

# Ensure the path to save the images exists
os.makedirs(path_to_save , exist_ok = True)

# Create list of tensor to save all the sample
if conversion_type == 1 or conversion_type == 2 : dataset_tensor = []

# Read the labels
with open(f'./data/ADNI_Labels/{dataset_name}_int.json') as fp: subj_to_label_int = json.load(fp)
with open(f'./data/ADNI_Labels/{dataset_name}_str.json') as fp: subj_to_label_str = json.load(fp)

# torchvision function to rescale images
if apply_resize :
    preprocess_function = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resize_size, resize_size))
    ])
else :
    preprocess_function = None

def get_conversion_factor(pixel_max_value : int) :
    """
    Get the factor to scale the pixel value between 0 and 4095
    """
    if pixel_max_value >= 0 and pixel_max_value <= 4095 :
        return 1
    elif pixel_max_value > 4095 and pixel_max_value <= 8191 :
        return 2
    elif pixel_max_value > 8191 and pixel_max_value <= 16383 :
        return 4
    elif pixel_max_value > 16383 and pixel_max_value <= 32768 :
        return 8
    elif pixel_max_value > 32768 and pixel_max_value <= 65535 :
        return 16
    else :
        raise ValueError(f"max pixel_max_value outside valid range. The value must be between 0 and 65535. The current value is {pixel_max_value}.")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert the data to images

n_element_to_print = int(len(list_files) / 252)

for i in range(len(list_files)) :
    if i % n_element_to_print == 0 : print(f"Processing file {i}/{len(list_files)}\t({round(i / len(list_files) * 100, 2)}%)")

    # Get file path
    file_path = list_files[i]

    # Create file path for the image
    file_path_decomposition = file_path.split('/')
    
    # Get info
    file_name = file_path_decomposition[-1]
    subj_id = file_path_decomposition[-5]
    date_aquisition = file_path_decomposition[-3]
    tmp_id = file_path_decomposition[-2]

    # Get folder for that specific recording
    path_specific_recording = file_path.split(file_name)[0]

    # Check if I have already converted that folder
    if path_specific_recording not in list_converted_folders :
        # Get all files for that specific recording
        list_files_specific_recording = support_dataset.get_all_files_from_path(path_specific_recording, filetype_filter = 'dcm')
        
        # Create folders to save the data
        recording_id = f'{subj_id}___{date_aquisition}___{tmp_id}'

        # Get indices of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        # Sort indices to convert map in the correct order
        idx_to_use = np.argsort(idx_images_list)

        tmp_image = []
        tmp_info = []
        for j in range(len(list_files_specific_recording)) :
            idx = idx_to_use[j]
            file_name = list_files_specific_recording[idx].split('/')[-1].split('.')[0]
            file_path_save = f'{path_to_save}{recording_id}_{file_name}.pt'

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[idx])

            # Get the pixel data
            pixel_data = data.pixel_array

            # (OPTIONAL) Rescale in range 0-4095
            if apply_pixel_rescaling  :
                rescale_factor = get_conversion_factor(pixel_data.max())
                pixel_data = pixel_data / rescale_factor

            # (OPTIONAL) Resize
            if preprocess_function is not None : 
                pixel_data = torch.from_numpy(pixel_data).squeeze() # The squeeze here is necessary because some images have 1 extra dimension while other do not
                pixel_data = preprocess_function(pixel_data.unsqueeze(0)).squeeze().type(torch.uint16)
            else :
                pixel_data = torch.from_numpy(pixel_data).type(torch.uint16).squeeze()
            
            tmp_image.append(pixel_data.squeeze())
        
        # Convert to numpy array
        tmp_image = np.asarray(tmp_image)

        # Save dicom info for the sample
        tmp_info.append(subj_id)
        tmp_info.append(subj_to_label_int[subj_id])
        tmp_info.append(subj_to_label_str[subj_id])
        tmp_info.append(data.BitsAllocated)
        tmp_info.append(data.BitsStored)
        tmp_info.append(data.HighBit)
        info_array.append(tmp_info)

        # Save image
        if conversion_type == 0 or conversion_type == 2 :
            try :
                torch.save(tmp_image, file_path_save)
            except Exception as e :
                print(f"Error saving image {i}:\n{e}")

        if conversion_type == 1 or conversion_type == 2 : dataset_tensor.append(tmp_image)

        list_converted_folders.append(path_specific_recording)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Save info (and single dataset tensor)
info_dataframe = pd.DataFrame(data = info_array, columns = ['subj_id', 'labels_int', 'labels_str', 'bits_allocated', 'bits_stored', 'high_bit'])
info_dataframe.to_csv(f'{path_to_save}info_dataframe.csv')
if conversion_type == 1 or conversion_type == 2 :
    for i in range(len(dataset_tensor)) : dataset_tensor[i] = dataset_tensor[i].squeeze()
    dataset_tensor = torch.from_numpy(np.asarray(dataset_tensor))
    dataset_tensor_path = f'{path_to_save}dataset_tensor'
    if preprocess_function is not None : dataset_tensor_path += f'___{resize_size}_resize'
    if apply_pixel_rescaling : dataset_tensor_path += '___pixel_rescaling'
    dataset_tensor_path += '.pt'
    torch.save(dataset_tensor, dataset_tensor_path)
    
