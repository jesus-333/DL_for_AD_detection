"""
Works in a similar way to the single file version but compute the stats for each sample, i.e. collection of slice of a single recording
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_48_slice_3"
path_all_data = f"./data/{dataset_name}/"
z_value = int(dataset_name.split('_')[4])

path_to_save = f"./data/info/{dataset_name}_samples/"
os.makedirs(path_to_save, exist_ok=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')
list_converted_folders = []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Variable to save results

max_per_slice_per_sample = [] # This will become a 2D array where each element is the max for that specific slice of that specific sample

bits_allocated_per_sample = []
bits_stored_per_sample    = []
bits_high_per_sample      = []

folder_to_with_problem = []

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

        # Get order of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        tmp_max_per_slice = []
        tmp_bits_allocated = []
        tmp_bits_stored = []
        tmp_bits_high = []
        for j in range(len(list_files_specific_recording)) :
            idx = idx_images_list[j]
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[j])

            # Get and save max value
            tmp_max_per_slice.append(data.pixel_array.max())

            # Get and save bits inforation
            tmp_bits_allocated.append(data.BitsAllocated)
            tmp_bits_stored.append(data.BitsStored)
            tmp_bits_high.append(data.HighBit)
        
        # Save information per sample
        max_per_slice_per_sample.append(tmp_max_per_slice)
        bits_allocated_per_sample.append(tmp_bits_allocated)
        bits_stored_per_sample.append(tmp_bits_stored)
        bits_high_per_sample.append(tmp_bits_high)

        if len(set(tmp_bits_allocated)) > 1 :
            folder_to_with_problem.append(path_specific_recording)
        
        # Add the folder to the list of converted folders
        list_converted_folders.append(path_specific_recording)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get idx of the samples that have the same number of slices
idx = np.array([len(x) for x in max_per_slice_per_sample]) == z_value
max_per_slice_per_sample = np.array([x for i, x in enumerate(max_per_slice_per_sample) if idx[i]])
bits_allocated_per_sample = np.array([x for i, x in enumerate(bits_allocated_per_sample) if idx[i]])
bits_stored_per_sample = np.array([x for i, x in enumerate(bits_stored_per_sample) if idx[i]])
bits_high_per_sample = np.array([x for i, x in enumerate(bits_high_per_sample) if idx[i]])

# Save the data
np.save(f"{path_to_save}max_per_slice_per_sample.npy", max_per_slice_per_sample)
np.save(f"{path_to_save}bits_allocated_per_sample.npy", bits_allocated_per_sample)
np.save(f"{path_to_save}bits_stored_per_sample.npy", bits_stored_per_sample)
np.save(f"{path_to_save}bits_high_per_sample.npy", bits_high_per_sample)
np.save(f"{path_to_save}list_analyzed_folders.npy", np.array(list_converted_folders))

