"""
It is similar to V4_2, but instead of saving each sample as separate list of images, it save them as a singular 3D tensor/array.
The 3D tensor array are created with the numpy library. A similar script, but with torch instead of numpy is the V4_3_torch.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import numpy as np
import os
import pandas as pd
import pickle
import pydicom as dicom

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_48_slice_3"
path_all_data = f"./data/{dataset_name}/"
path_to_save = f"./data/{dataset_name}_png_V4_3/"

conversion_type = 1 # 0 = 1 3D tensor for each sample (i.e. 1 sample, 1 file), 1 = A list of 3D tensors for all the dataset (i.e. all the dataset is saved in a list of tensors), 2 = both

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert the data to images

n_element_to_print = 200

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
        file_path_save = f'{path_to_save}{recording_id}/'

        # Get indices of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        # Sort indices to convert map in the correct order
        idx_to_use = np.argsort(idx_images_list)
    
        # Read dicom file
        tmp_image = []
        tmp_info = []
        for j in range(len(list_files_specific_recording)) :
            idx = idx_to_use[j]
            file_name = list_files_specific_recording[idx].split('/')[-1].split('.')[0]
            file_path_save = f'{path_to_save}{recording_id}_{file_name}.npy'

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[idx])

            # Get the pixel data
            pixel_data = data.pixel_array
            
            tmp_image.append(pixel_data)
        
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
                np.save(file_path_save, tmp_image)
            except Exception as e :
                print(f"Error saving image {i}:\n{e}")

        if conversion_type == 1 or conversion_type == 2 : dataset_tensor.append(tmp_image)

        list_converted_folders.append(path_specific_recording)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Save info (and single dataset tensor)
info_dataframe = pd.DataFrame(data = info_array, columns = ['subj_id', 'labels_int', 'labels_str', 'bits_allocated', 'bits_stored', 'high_bit'])
info_dataframe.to_csv(f'{path_to_save}info_dataframe.csv')
info_array = np.asarray(info_array)
np.save(f'{path_to_save}info_array.npy', info_array)
if conversion_type == 1 or conversion_type == 2 :
    with open(f'{path_to_save}dataset_tensor.pkl', 'wb') as f:
        pickle.dump(dataset_tensor, f)
    
