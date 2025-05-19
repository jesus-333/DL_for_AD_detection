"""
From the ADNI dataset it is possible to download only the 2D MRI images, saved in dcm format.
This script takes the path, find all dcm files and converts all the dcm files to png files.
This version is a variation of V4_1. Each recording will have its own folder but the subj will be at the same level. The name of the subj is preserved in the folder name (this will be useful later to recover the label).
N.B. Note that this script is created using my subfolder structure as reference. Note also that the path to the dataset is in the form "./xxx/", i.e. it start with the dot

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
E.g. of the final results
- {recording_id}_subj 1 :
    - img_0.png
    - img_1.png
    ...
    -img_n.png
- {recording_id}_subj 1 :
    - img_0.png
    - img_1.png
    ...
    -img_n.png 
- {recording_id}_subj n :
    - img_0.png
    - img_1.png
    ...
    -img_n.png 
etc
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import numpy as np
import pydicom as dicom
import matplotlib
import matplotlib.pyplot as plt # It is not used but it is necessary to use matplotlib.image() for the conversion
import cv2 as cv
import PIL

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_48_slice_3"
path_all_data = f"./data/{dataset_name}/"

# Conversion settigns
conversion_method = "matplotlib"

if conversion_method == 'numpy' :
    path_to_save = f"./data/{dataset_name}_npy_V4_2/"
else :
    path_to_save = f"./data/{dataset_name}_png_V4_2/"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')
list_converted_folders = []

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

        # Ensure the path to save the images exists
        os.makedirs(file_path_save , exist_ok = True)

        # Get order of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        # Sort indices to convert map in the correct order
        idx_to_use = np.argsort(idx_images_list)

        for j in range(len(list_files_specific_recording)) :
            # idx = idx_images_list[j]
            idx = idx_to_use[j]
            file_name = list_files_specific_recording[idx].split('/')[-1].split('.')[0]
            file_path_save = f'{path_to_save}{recording_id}/{idx}_{file_name}.png'

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[idx])

            # Get the pixel data
            pixel_data = data.pixel_array

            # Save image
            try :
                if conversion_method == 'matplotlib' :
                    matplotlib.image.imsave(file_path_save, pixel_data, cmap = 'gray')
                elif conversion_method == 'cv' :
                    cv.imwrite(file_path_save, pixel_data)
                elif conversion_method == 'PIL' :
                    image = PIL.Image.fromarray(pixel_data)
                    image.save(file_path_save)
                elif conversion_method == 'numpy' :
                    # np.save(file_path_save.replace('.png', '.npy'), pixel_data / pixel_data.max())
                    np.save(file_path_save.replace('.png', '.npy'), pixel_data)
                else :
                    raise ValueError(f"Conversion method {conversion_method} not recognized. Use 'matplotlib', 'cv', 'PIL', or 'numpy'.")
            except Exception as e :
                print(f"Error saving image {i}:\n{e}")

        list_converted_folders.append(path_specific_recording)
