"""
From the ADNI dataset it is possible to download only the 2D MRI images, saved in dcm format.
This script takes the path, find all dcm files and converts all the dcm files to png files.
This version is a combination of V3 and V4. Each subject will have his/her own folder and inside each recording will have its own folder.
N.B. Note that this script is created using my subfolder structure as reference. Note also that the path to the dataset is in the form "./xxx/", i.e. it start with the dot

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
E.g. of the final results
- subj 1 :
    - Recording_1 :
        - img_0.png
        - img_1.png
        ...
        -img_n.png
    - Recording_2:
        - img_0.png
        - img_1.png
        ...
        -img_n.png
- subj 2 :
    - Recording_1:
        - img_0.png
        - img_1.png
        ...
        -img_n.png
etc
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import numpy as np
import pydicom as dicom
import matplotlib
import cv2 as cv
import PIL

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_preprocess_collection_all"
path_data = f"./data/{dataset_name}/"
path_to_save = f"./data/{dataset_name}_png_V4_1/"

# Conversion settigns
conversion_method = "matplotlib"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert the data to images

for i in range(len(list_files)) :
    print(f"Processing file {i}/{len(list_files)}\t({round(i / len(list_files) * 100, 2)}%)")

    # Get file path
    file_path = list_files[i]

    # Create file path for the image
    file_path_decomposition = file_path.split('/')

    subj_id = file_path_decomposition[-5]
    
    date_aquisition = file_path_decomposition[-3]
    tmp_id = file_path_decomposition[-2]
    recording_id = f'{date_aquisition}_{tmp_id}'

    file_name = file_path_decomposition[-1].split('.')[0] + '.png'

    file_path_save = f'{path_to_save}{subj_id}/{recording_id}/'

    # Ensure the path to save the images exists
    os.makedirs(file_path_save , exist_ok = True)
    file_path_save += file_name

    # Load the data
    data = dicom.dcmread(file_path)

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
        else :
            raise ValueError(f"Conversion method {conversion_method} not recognized. Use 'matplotlib', 'cv' or 'PIL'.")
    except Exception as e :
        print(f"Error saving image {i}:\n{e}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Code used for ad-hoc analysis
# idx_analysis = [5516, 5517, 5561, 5562, 5563]
# for i in range(len(idx_analysis)) :
#
#     file_path = list_files[idx_analysis[i]]
#
#     # Load the data
#     data = dicom.dcmread(file_path)
#
#     # Get the pixel data
#     pixel_data = data.pixel_array
#
#     hist, bins = np.histogram(pixel_data.flatten(), bins = 256)
#     bins = bins[:-1]
#
#     n_el = len(pixel_data.flatten())
#
#     thresh = 90
#     value_1 = np.sum(hist[bins <= 6])
#     value_2 = np.sum(hist[np.logical_and(bins > 6, bins < thresh)])
#     value_3 = np.sum(hist[bins >= thresh])
#
#     print(f"Image {idx_analysis[i]}:")
#     print(f"\tValue 1: {value_1} ({round(value_1/n_el * 100, 2)}%)")
#     print(f"\tValue 2: {value_2} ({round(value_2/n_el * 100, 2)}%)")
#     print(f"\tValue 3: {value_3} ({round(value_3/n_el * 100, 2)}%)\n")
