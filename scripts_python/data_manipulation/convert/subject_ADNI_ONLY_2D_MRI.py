"""
From the ADNI dataset it is possible to download only the 2D MRI images, saved in dcm format.
This script takes the path to a single subject from the ADNI dataset, and converts all the dcm files to png files.
The folder structure is preserved, so the images are saved in the same subfolder as the original dcm files.
Notes that this files is created to work with my specific directory structure, so it may need to be adapted to work in other cases.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import numpy as np
import pydicom as dicom
import cv2 as cv

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_subject = './data/ADNI_MRI_2D/MCI/002_S_0729/'
path_to_save = 'TMP/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get all folders from the path
list_folders = os.listdir(path_subject)
list_folders = [path_subject + folder + '/' for folder in list_folders if os.path.isdir(path_subject + folder)]

for path_data in list_folders :
    print(path_data)
    # Get all the files from the path
    list_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Convert the data to images

    for i in range(len(list_files)) :
        print(f'\tProcessing file {i}/{len(list_files)}')
        
        file_path = list_files[i]

        # Load the data
        data = dicom.dcmread(file_path)

        # Get the pixel data
        pixel_data = data.pixel_array
        pixel_data[pixel_data < 0] = 0
        pixel_data[pixel_data > 255] = 255

        # Ensure the path to save the images exists
        image_type = file_path.split('/')[5]
        tmp_path_save = path_to_save + f'{image_type}/'
        os.makedirs(tmp_path_save , exist_ok = True)

        # Save image
        cv.imwrite(tmp_path_save + f'img_{i}.png', pixel_data)


