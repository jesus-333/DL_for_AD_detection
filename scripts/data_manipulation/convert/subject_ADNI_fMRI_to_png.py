"""
Convert all the data, from dcm file format to png file format, for a single subject inside the ADNI dataset.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import numpy as np
import pydicom as dicom
import cv2 as cv

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_data = 'data/ADNI_3_fMRI/MCI/002_S_1155/Resting_State_fMRI/2011-12-22_09_35_38.0/I274147/'
path_to_save = 'TMP/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get all the files from the path
list_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')

# Ensure the path to save the images exists
os.makedirs(path_to_save, exist_ok = True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Convert the data to images

for i in range(len(list_files)) :
    print(f'Processing file {i}/{len(list_files)}')
    
    file_path = list_files[i]

    # Load the data
    data = dicom.dcmread(file_path)

    # Get the pixel data
    pixel_data = data.pixel_array
    pixel_data[pixel_data < 0] = 0
    pixel_data[pixel_data > 255] = 255

    # Save the image
    cv.imwrite(path_to_save + f'img_{i}.png', pixel_data)

