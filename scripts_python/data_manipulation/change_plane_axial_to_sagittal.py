"""
NOT WORKING
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import numpy as np
import pydicom as dicom
import matplotlib
import matplotlib.pyplot as plt

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_44_slice_4"
path_all_data = f"./data/{dataset_name}/"
path_to_save = f"./data/{dataset_name.replace('axial', 'sagittal')}_png_V4_2/"

normalize_type = 0 # 0 ---> slice per slice, 1 ---> all 3D tensor together

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')
list_converted_folders = []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert the data to images

for i in range(len(list_files)) :
    print(f"Processing file {i}/{len(list_files)}\t({round(i / len(list_files) * 100, 2)}%)")

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

        # Ensure the path to save the images exists
        os.makedirs(file_path_save , exist_ok = True)

        # Get order of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        tmp_image = []
        for j in range(len(list_files_specific_recording)) :
            idx = idx_images_list[j]
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[j])

            # Get the pixel data
            pixel_data = data.pixel_array
            
            # Normalize in 0-1 range
            if normalize_type == 0 : pixel_data = pixel_data / pixel_data.max()

            tmp_image.append(pixel_data)
    
        tmp_image = np.asarray(tmp_image)
        if normalize_type == 1 : tmp_image /= tmp_image.max()

        for j in range(tmp_image.shape[1]) :
            pixel_data = tmp_image[:, j, :]
            file_path_save = f'{path_to_save}{recording_id}/{j}.png'

            # Save image
            try :
                # np.save(file_path_save.replace('.png', '.npy'), tmp_image)
                matplotlib.image.imsave(file_path_save, pixel_data, cmap = 'gray')
            except Exception as e :
                print(f"Error saving image: {e}")

        list_converted_folders.append(path_specific_recording)
