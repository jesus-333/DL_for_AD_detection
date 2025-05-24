"""
Read all the dicom file inside a dataset and get the max value of pixel, bits allocated, bits stored and high bit
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

path_to_save = f"./data/info/{dataset_name}_single_files/"
os.makedirs(path_to_save, exist_ok=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get all the files and filter only the dcm files
list_files = support_dataset.get_all_files_from_path(path_all_data, filetype_filter = 'dcm')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert the data to images
n_element_to_print = int(len(list_files) / 252)

bits_allocated_list = np.zeros(len(list_files))
bits_stored_list    = np.zeros(len(list_files))
bits_high_list      = np.zeros(len(list_files))
max_value_list      = np.zeros(len(list_files))

for i in range(len(list_files)) :
    if i % n_element_to_print == 0 : print(f"Processing file {i}/{len(list_files)}\t({round(i / len(list_files) * 100, 2)}%)")

    # Get file 
    data = dicom.dcmread(list_files[i])
    
    # Get info for the specific file
    bits_allocated_list[i] = data.BitsAllocated
    bits_stored_list[i]    = data.BitsStored
    bits_high_list[i]      = data.HighBit

    try :
        max_value_list[i] = data.LargestImagePixelValue
    except :
        # print(f"Problem with max value with image {list_files[i]}")
        max_value_list[i] = data.pixel_array.max()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Print info about the images

def print_info(array_to_analyze) :
    unique_values_list = np.unique_values(array_to_analyze)

    for unique_value in unique_values_list :
        print(f"Value {unique_value} : {np.sum(array_to_analyze == unique_value)}")

print("Bits Allocated")
print_info(bits_allocated_list)

print("\nBits Stored")
print_info(bits_stored_list)

print("\nBits High")
print_info(bits_high_list)


# Save the data
np.save(f"{path_to_save}bits_allocated.npy", bits_allocated_list)
np.save(f"{path_to_save}bits_stored.npy", bits_stored_list)
np.save(f"{path_to_save}bits_high.npy", bits_high_list)
np.save(f"{path_to_save}max_value.npy", max_value_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# idx = np.logical_and(max_value_list > 4095, bits_high_list == 15)
# # idx = bits_high_list == 15
# random_file = np.random.choice(np.asarray(list_files)[idx], replace = False)
# tmp_data_dcm = dicom.dcmread(random_file)
#
# tmp_data_dcm = dicom.dcmread(aaa[15])
# tmp_data = tmp_data_dcm.pixel_array
#
# fig, axs = plt.subplots(2, 2, figsize = (10, 10))
#
# axs[0, 0].imshow(tmp_data, cmap = 'gray')
# axs[0, 1].hist(tmp_data.flatten(), bins = 100, label = f"{tmp_data_dcm.HighBit}, {tmp_data.max()}")
# axs[0, 1].grid(True)
# axs[0, 1].legend()
#
# tmp_data_filtered = tmp_data
# tmp_data_filtered[tmp_data_filtered > 4095] = 4095
# axs[1, 0].imshow(tmp_data_filtered, cmap = 'gray')
# axs[1, 1].hist(tmp_data_filtered.flatten(), bins = 100, label = f"{tmp_data_dcm.HighBit}, {tmp_data_filtered.max()}")
# axs[1, 1].grid(True)
# axs[1, 1].legend()
#
# fig.tight_layout()
# plt.show()
