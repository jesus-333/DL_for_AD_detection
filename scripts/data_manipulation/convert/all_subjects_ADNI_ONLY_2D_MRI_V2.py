"""
From the ADNI dataset it is possible to download only the 2D MRI images, saved in dcm format.
This script takes the path, find all dcm files and converts all the dcm files to png files.
This is similar to the single subject version but in this case the structure of the subfolder IS NOT PRESERVED, so all images will be saved in the path specified by path_to_save
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

type_data = 'AD'
dataset_name = 'ADNI_axial_PD_150_TSE'
dataset_folder = f'./data/{dataset_name}'

path_data = os.path.join(dataset_folder, type_data + '/')
path_to_save = os.path.join(dataset_folder + '_png', type_data + '/')

# Conversion settigns
conversion_method = 'matplotlib'
apply_pixel_filter = False
minumum_size = 100 
apply_hist_filter = True

idx_to_skip_1 = np.arange(31798, 31902)
idx_to_skip_1 = []
idx_files_ignored_with_hist = []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')

# Ensure the path to save the images exists
os.makedirs(path_to_save , exist_ok = True)
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Convert the data to images

for i in range(len(list_files)) :
    print(f"Processing file {i}/{len(list_files)}\t({round(i/len(list_files) * 100, 2)}%)")
    
    if i in idx_to_skip_1 :
        print(f"\tSkipping file {i}")

    file_path = list_files[i]

    # Load the data
    data = dicom.dcmread(file_path)

    # Get the pixel data
    pixel_data = data.pixel_array
    if apply_pixel_filter :
        pixel_data[pixel_data < 0] = 0
        pixel_data[pixel_data > 255] = 255

    if minumum_size > 0 :
        if pixel_data.shape[0] < minumum_size or pixel_data.shape[1] < minumum_size :
            print(f"\tImage too small. Skipping {file_path}")
            continue

    if apply_hist_filter :
        hist, bins = np.histogram(pixel_data.flatten(), bins = 256)
        bins = bins[:-1]

        # This is simply an empircal solution that seems to (mostly) work.
        # I chose this values after plotting somes histograms of normal iamges and corrupted images.
        if np.sum(hist[bins > 1500]) > 20000  :
            idx_files_ignored_with_hist.append(i)
            print(f"\tImage skipped due to hist values (1). Skipping {file_path}")
            continue
        n_el = len(pixel_data.flatten())
        value_2 = np.sum(hist[np.logical_and(bins > 6, bins < 90)]) / n_el
        if value_2 >= 0.65 :
            idx_files_ignored_with_hist.append(i)
            print(f"\tImage skipped due to hist values (2). Skipping {file_path}")
            continue

    # Save image
    try :
        if conversion_method == 'matplotlib' :
            matplotlib.image.imsave(path_to_save + f'img_{i}.png', pixel_data, cmap = 'gray')
        elif conversion_method == 'cv' : 
                cv.imwrite(path_to_save + f'img_{i}.png', pixel_data)
        elif conversion_method == 'PIL' :
            image = PIL.Image.fromarray(pixel_data)
            image.save(path_to_save + f'img_{i}.png')
        else :
            raise ValueError(f"Conversion method {conversion_method} not recognized. Use 'matplotlib', 'cv' or 'PIL'.")
    except Exception as e :
        print(f"Error saving image {i}:\n{e}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

print("Saving histograms of the images that were ignored due to hist values")
path_to_save_hist = f'./TMP_{dataset_name}/{type_data}/'
os.makedirs(path_to_save_hist, exist_ok = True)
for i in range(len(idx_files_ignored_with_hist)) :
    idx = idx_files_ignored_with_hist[i]
    print(f"Processing file {i}/{len(idx_files_ignored_with_hist)}\t({round(i/len(idx_files_ignored_with_hist) * 100, 2)}%)")
    try : 
        file_path = list_files[idx]
        data = dicom.dcmread(file_path)
        pixel_data = data.pixel_array
        fig, ax = plt.subplots(1, 2, figsize = (15, 10))
        ax[0].imshow(pixel_data, cmap = 'gray')
        ax[0].set_title('Original image')
        ax[1].hist(pixel_data.flatten(), bins = 256, histtype = 'step', color = 'black')
        ax[1].set_title('Histogram of pixel values')
        hist, bins = np.histogram(pixel_data.flatten(), bins = 256)
        fig.tight_layout()
        fig.savefig(path_to_save_hist + f'img_{idx}.png')
        plt.close(fig)
    except Exception as e :
        print(f"Error saving image {file_path}:\n{e}")


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
#
