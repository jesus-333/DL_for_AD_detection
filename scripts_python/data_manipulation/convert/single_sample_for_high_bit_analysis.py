"""
Take a sample with an high bit of 15 and convert it to png two times.
The first time simply convert the data as they are.
The second cap the pixel with value above 4095 to 4095
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt

from src.dataset import support_dataset
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_48_slice_3"
path_all_data = f"./data/{dataset_name}/"
z_value = int(dataset_name.split('_')[4])

path_with_info = f"./data/info/{dataset_name}_samples/"
path_save = f"./data/info/{dataset_name}_comparison_high_bit/"
os.makedirs(path_save, exist_ok=True)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load info

max_per_slice_per_sample = np.load(f"{path_with_info}max_per_slice_per_sample.npy")
bits_allocated_per_sample = np.load(f"{path_with_info}bits_allocated_per_sample.npy")
bits_stored_per_sample = np.load(f"{path_with_info}bits_stored_per_sample.npy")
bits_high_per_sample = np.load(f"{path_with_info}bits_high_per_sample.npy")
list_analyzed_folders = np.load(f"{path_with_info}list_analyzed_folders.npy")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Random sample a folder
idx_high_bit_15 = np.logical_and(bits_high_per_sample[:, 0] == 15, np.sum(max_per_slice_per_sample > 4095, axis = 1))
list_possible_folders = list_analyzed_folders[idx_high_bit_15]
folder_path = np.random.choice(list_possible_folders, 1)[0]

list_files_specific_folder = support_dataset.get_all_files_from_path(folder_path, filetype_filter = 'dcm')

for i in range(len(list_files_specific_folder)) :
    file_path = list_files_specific_folder[i]

    data = dicom.dcmread(file_path)
    
    pixel_data = data.pixel_array
    print(f"max pixel value: {np.max(pixel_data)}")
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(pixel_data, cmap = 'gray')
    axs[0].set_title(f"Original image {i}")

    # Cap the pixel values
    capped_pixel_data = pixel_data.copy()
    capped_pixel_data[capped_pixel_data > 2**13] = 2**13
    axs[1].imshow(capped_pixel_data, cmap = 'gray')
    axs[1].set_title(f"Capped image {i}")
    
    # Rescale between 0 and 4095
    rescaled_pixel_data = pixel_data.copy()
    rescaled_pixel_data = (rescaled_pixel_data / np.max(rescaled_pixel_data)) * 4095
    # rescaled_pixel_data = np.array(rescaled_pixel_data / 8, dtype = int)
    axs[2].imshow(rescaled_pixel_data, cmap = 'gray')
    axs[2].set_title(f"Rescaled image {i}")

    # Save the figure 
    fig.tight_layout()
    idx_file = file_path.split('/')[-1].split('_')[-3]
    plt.savefig(f"{path_save}img_{idx_file}.png")
    plt.close()
