"""
Convert nii files to numpy arrays and save them in a folder. Also, save a dictionary and a dataframe with information about the dataset.

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

    # Get nii data
    img = support_dataset.get_nii_data(file_path)

    # Create file path for the image
    file_path_decomposition = file_path.split('/')
    
    # Get info
    file_name = file_path_decomposition[-1]
    subj_id = file_path_decomposition[-5]
    date_aquisition = file_path_decomposition[-3]
    tmp_id = file_path_decomposition[-2]

    # Create file name for the recording
    file_name_save = f'{subj_id}___{date_aquisition}___{tmp_id}___{dataset_name}.npy'
    file_path_save = f'{path_to_save}{file_name_save}'

    # Temporary variable to store information about the sample
    tmp_image = img.get_fdata()
    tmp_info  = []
    tmp_dict  = dict()

    # Save nii info for the sample (dataframe)
    tmp_info.append(file_name_save)
    tmp_info.append(subj_id)
    tmp_info.append(subj_to_label_int[subj_id])
    tmp_info.append(subj_to_label_str[subj_id])
    tmp_info.append(dataset_name)
    tmp_info.append(len(tmp_image))
    tmp_info.append(tmp_image.shape[1])
    tmp_info.append(tmp_image.shape[2])
    tmp_info.append(-1) # BitsAllocated. Note that I get this information only when I work with dicom files because I get it from dicom metadata (see dicom script). In order to have the same dataframe from both scripts I put -1 for the nii script.
    tmp_info.append(-1) # BitStored. See comment for BitsAllocated
    tmp_info.append(-1) # HighBit. See comment for BitsAllocated
    tmp_info = np.asarray(tmp_info)
    dataset_info_csv = pd.concat([dataset_info_csv, pd.DataFrame(tmp_info.reshape(1, -1), columns = columns)], ignore_index = True)

    # Save nii info for the sample (dictionary)
    tmp_dict['subj_id'] = subj_id
    tmp_dict['label_int'] = subj_to_label_int[subj_id]
    tmp_dict['label_str'] = subj_to_label_str[subj_id]
    tmp_dict['original_dataset'] = dataset_name
    tmp_dict['original_shape'] = img.shpae
    tmp_dict['z_matrix'] = len(tmp_image)
    tmp_dict['bits_allocated'] = -1 # See comment for BitsAllocated in the dataframe section.
    tmp_dict['bits_stored'] = -1    # See comment for BitStored in the dataframe section.
    tmp_dict['high_bit'] = -1       # See comment for HighBit in the dataframe section.
    dataset_info_dict[file_name_save] = tmp_dict

    # Save array
    np.save(file_path_save, tmp_image)
    
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
