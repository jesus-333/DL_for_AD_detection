"""
Since the files read from DCM have "strange" pixel values (i.e. the max is not 255 but some number in the order of thousands), here I plot a random DCM file, with its histogram.
I will also add the results of thresholding algorithm to that image to see the impact on how the image appears.

Results : inconclusive (for now)

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pydicom as dicom
import cv2 as cv
import matplotlib.pyplot as plt

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_39_slice_5"
path_all_data = f"./data/{dataset_name}/"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')
list_converted_folders = []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get image and compute threshold

# Get random image from the list
idx_random = np.random.randint(0, len(list_files))
data = dicom.dcmread(list_files[idx_random]).pixel_array

data_uint8 = data.astype(np.uint8)

# Otsu threshold
otsu_th_value, data_otsu_th = cv.threshold(data_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Canny algorithm detection
# Why use otsu threshold value in canny?
# https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
data_canny_edge = cv.Canny(data_uint8, otsu_th_value / 2, otsu_th_value)

img_to_plot = [data, data_uint8, data_otsu_th, data_canny_edge]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

figsize = (20, 12)
fig, axs = plt.subplots(1, 4, figsize = figsize)

# axs_original_image = axs[:, 0]
# axs_hist = axs[:, 1]
axs_original_image = axs

for i in range(4) :
    ax = axs_original_image[i]
    ax.imshow(img_to_plot[i], cmap = 'gray')

fig.tight_layout()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
