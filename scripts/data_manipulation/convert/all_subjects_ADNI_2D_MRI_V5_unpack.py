"""
This script is similar to convert/all_subjects_ADNI_ONLY_2D_MRI_V4_2.py with the numpy settings.
So this file will convert all the dicom of a single scan in a single npy file (so 1 scan ---> 1 npy file)
Together with samples a dictionary and a csv will be created. The dictionary will have as key the file name and as values all the info about that specific scan (subject id, label, original dataset etc). The csv file will have the same information but in a table structure.
The main purpose is to convert all the data downloaded from the ADNI website in a format and structure which can be easily manipulated by the create_dataset.py script.

Note that no resize will be applied or any other preprocess will be applied to the image. All this operation will be performed by the create_dataset.py script.
The data pixel value will be between 0 and 4095 since most of the data are already in that range. Sample that have a different scaled will be rescaled in that range.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import numpy as np
import os
import pandas as pd
import pydicom as dicom

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_44_slice_4"
path_all_data = f"./data/{dataset_name}/"

path_to_save = "./data/ADNI_dicom_converted_axial/"
path_to_dataset_info_dict = f'{path_to_save}data_info.json'
path_to_dataset_info_csv = f'{path_to_save}data_info.csv'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')
list_converted_folders = dict()
info_array = []

# Ensure the path to save the images exists
os.makedirs(path_to_save , exist_ok = True)

# Read the labels
with open(f'./data/ADNI_Labels/{dataset_name}_int.json') as fp: subj_to_label_int = json.load(fp)
with open(f'./data/ADNI_Labels/{dataset_name}_str.json') as fp: subj_to_label_str = json.load(fp)

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

# Get dictionary with dataset info
dataset_info_dict = dict()

# Get datafram with dataset info
columns = ['file_name', 'subj_id', 'labels_int', 'labels_str', 'original_dataset', 'z_matrix', 'H', 'W', 'bits_allocated', 'bits_stored', 'high_bit']
dataset_info_csv = pd.DataFrame(data = None, columns = columns)

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
        
        # Create file name for the recording
        file_name_save = f'{subj_id}___{date_aquisition}___{tmp_id}___{dataset_name}.npy'
        file_path_save = f'{path_to_save}{file_name_save}'

        # Get indices of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        # Sort indices to convert map in the correct order
        idx_to_use = np.argsort(idx_images_list)

        # Temporary variable to store information about the sample
        tmp_image = []
        tmp_info  = []
        tmp_dict  = dict()

        # Get all slices for the sample
        for j in range(len(list_files_specific_recording)) :
            idx = idx_to_use[j]

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[idx])

            # Get the pixel data
            pixel_data = data.pixel_array.squeeze()

            # Rescale in range 0-4095
            rescale_factor = get_conversion_factor(pixel_data.max())
            pixel_data = pixel_data / rescale_factor

            # Cast to int
            pixel_data = pixel_data.astype(np.uint16)
            
            # Append to 3D array
            tmp_image.append(pixel_data.squeeze())

        # Convert to numpy array
        tmp_image = np.asarray(tmp_image).squeeze()

        # Save dicom info for the sample (dataframe)
        tmp_info.append(file_name_save)
        tmp_info.append(subj_id)
        tmp_info.append(subj_to_label_int[subj_id])
        tmp_info.append(subj_to_label_str[subj_id])
        tmp_info.append(dataset_name)
        tmp_info.append(len(tmp_image))
        tmp_info.append(tmp_image.shape[1])
        tmp_info.append(tmp_image.shape[2])
        tmp_info.append(data.BitsAllocated)
        tmp_info.append(data.BitsStored)
        tmp_info.append(data.HighBit)
        tmp_info = np.asarray(tmp_info)
        dataset_info_csv = pd.concat([dataset_info_csv, pd.DataFrame(tmp_info.reshape(1, -1), columns = columns)], ignore_index = True)

        # Save dicom info for the sample (dictionary)
        tmp_dict['subj_id'] = subj_id
        tmp_dict['label_int'] = subj_to_label_int[subj_id]
        tmp_dict['label_str'] = subj_to_label_str[subj_id]
        tmp_dict['original_dataset'] = dataset_name
        tmp_dict['original_shape'] = pixel_data.shape
        tmp_dict['z_matrix'] = len(tmp_image)
        tmp_dict['bits_allocated'] = data.BitsAllocated
        tmp_dict['bits_stored'] = data.BitsStored
        tmp_dict['high_bit'] = data.HighBit
        dataset_info_dict[file_name_save] = tmp_dict

        # Save array
        np.save(file_path_save, tmp_image)
        
        # Add folder to list of analyzed folder.
        # Note that I'm interested only in the key and not in the value fied. I use the dictionary as a lookupt table.
        list_converted_folders[path_specific_recording] = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save dataset info

# Dictionary
try :
    with open(path_to_dataset_info_dict) as f: dataset_info_dict_already_saved = json.load(f)
    new_dataset_info_dict = dataset_info_dict | dataset_info_dict_already_saved
    with open(path_to_dataset_info_dict, 'w') as f: json.dump(new_dataset_info_dict, f)
except :
    print(f"File {path_to_dataset_info_dict} not found. A new dictionary for dataset info will be created")
    with open(path_to_dataset_info_dict, 'w') as f: json.dump(dataset_info_dict, f)

# Dataframe
try :
    dataset_info_csv_already_saved = pd.read_csv(path_to_dataset_info_csv)
    new_dataset_info_csv = pd.concat([dataset_info_csv_already_saved, dataset_info_csv], ignore_index = True)
    new_dataset_info_csv.to_csv(path_to_dataset_info_csv, index = False)
except :
    print(f"File {path_to_dataset_info_csv} not found. A new dataframe for dataset info will be created")
    dataset_info_csv.to_csv(path_to_dataset_info_csv, index = False)
