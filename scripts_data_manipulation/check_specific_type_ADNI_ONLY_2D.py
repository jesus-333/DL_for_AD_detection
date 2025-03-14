"""
Check specific type of acquisition for the dcm ADNI dataset

I noticed that sometimes with the deafualt conversion the image is completely black.
With this script I want to check/visualize these images and understand if there is a problem during the conversion. 
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import numpy as np
import pydicom as dicom
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import PIL

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_dataset = './data/ADNI_MRI_2D/MCI/'
path_to_save = 'TMP/'

type_to_check = 'Double_TSE'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get all the subject
list_subjects = os.listdir(path_dataset)
list_subjects = [path_dataset + subject + '/' for subject in list_subjects if os.path.isdir(path_dataset + subject)]

list_path_to_analyze = []

for path_subject in list_subjects :
    # print(path_subject)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Get all folders from the path
    list_folders = os.listdir(path_subject)
    list_folders = [path_subject + folder + '/' for folder in list_folders if os.path.isdir(path_subject + folder)]

    for path_data in list_folders :
        # print("\t", path_data)
        # Get all the files from the path
        list_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Convert the data to images

        for i in range(len(list_files)) :
            # print(f'\t\tProcessing file {i}/{len(list_files)}')
            
            file_path = list_files[i]

            if type_to_check in file_path : list_path_to_analyze.append(file_path)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def visualize_image(path : str) :
    # Load the data
    data = dicom.dcmread(path)

    # Get the pixel data
    pixel_data = data.pixel_array

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(pixel_data, cmap = 'gray')
    axs[0, 1].hist(pixel_data.flatten(), bins = 256, color = 'black')

    filtered_pixed_data = np.copy(pixel_data)
    filtered_pixed_data[filtered_pixed_data < 0] = 0
    filtered_pixed_data[filtered_pixed_data > 255] = 255

    axs[1, 0].imshow(filtered_pixed_data, cmap = 'gray')
    axs[1, 1].hist(filtered_pixed_data.flatten(), bins = 256, histtype = 'step', color = 'black')

    fig.tight_layout()
    fig.show()

    return fig, axs, pixel_data, filtered_pixed_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_save_image_sample = '{}/0_Check_images/{}/'.format(path_to_save, type_to_check)

idx_random = np.random.randint(0, len(list_path_to_analyze))
fig, ax, pixel_data, filtered_pixed_data = visualize_image(list_path_to_analyze[idx_random])

os.makedirs(path_save_image_sample, exist_ok = True)

matplotlib.image.imsave(path_save_image_sample + 'img_matplotlib_raw.png', pixel_data, cmap = 'gray')
matplotlib.image.imsave(path_save_image_sample + 'img_matplotlib_filtered.png', filtered_pixed_data, cmap = 'gray')
cv.imwrite(path_save_image_sample + 'img_cv_raw.png', pixel_data)
cv.imwrite(path_save_image_sample + 'img_cv._filtered.png', filtered_pixed_data)
filtered_normalized = cv.normalize(filtered_pixed_data, dst = None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
cv.imwrite(path_save_image_sample + 'img_cv_normalized.png', filtered_normalized)
PIL.Image.fromarray(pixel_data).save(path_save_image_sample + 'img_pil_raw.png')
PIL.Image.fromarray(filtered_pixed_data).save(path_save_image_sample + 'img_pil_filtered.png')
PIL.Image.fromarray(filtered_normalized).save(path_save_image_sample + 'img_pil_normalized.png')


