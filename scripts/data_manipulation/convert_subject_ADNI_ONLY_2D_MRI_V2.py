"""
Similar to V1 but convert all subjects. The subject is not preserved but the type of image is preserved (e.g. T1, T2, etc).
Note that this is not used to prepare the dataset for training. 
I only want to see the differences between the various types of images.
Note that due to the ADNI folder structure, different subjects have different types of images, or, sometims, the same type of image is named differently.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import numpy as np
import pydicom as dicom
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import PIL

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_dataset = './data/ADNI_MRI_2D_Axial/MCI/'
path_to_save = 'TMP/'

conversion_method = 'matplotlib'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get all the subject
list_subjects = os.listdir(path_dataset)
list_subjects = [path_dataset + subject + '/' for subject in list_subjects if os.path.isdir(path_dataset + subject)]

j = 0
for idx_subject in range(len(list_subjects)) :
    path_subject = list_subjects[idx_subject]
    print(f"Processing subject {idx_subject + 1}/{len(list_subjects)} \t {path_subject}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Get all folders from the path
    list_folders = os.listdir(path_subject)
    list_folders = [path_subject + folder + '/' for folder in list_folders if os.path.isdir(path_subject + folder)]

    for path_data in list_folders :
        print("\t", path_data)
        # Get all the files from the path
        list_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Convert the data to images

        for i in range(len(list_files)) :
            # print(f'\t\tProcessing file {i}/{len(list_files)}')
            
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
            try :
                if conversion_method == 'matplotlib' :
                    matplotlib.image.imsave(tmp_path_save + f'img_{j}.png', pixel_data, cmap = 'gray')
                    # matplotlib.image.imsave(path_save_image_sample + 'img_matplotlib_raw.png', pixel_data, cmap = 'gray')
                    j += 1
                elif conversion_method == 'cv' : 
                        cv.imwrite(tmp_path_save + f'img_{j}.png', pixel_data)
                        j += 1
                elif conversion_method == 'PIL' :
                    image = PIL.Image.fromarray(pixel_data)
                    image.save(tmp_path_save + f'img_{j}.png')
                    j += 1
                else :
                    raise ValueError(f"Conversion method {conversion_method} not recognized. Use 'matplotlib', 'cv' or 'PIL'.")
            except :
                print(f"\tError in saving file {file_path}")
