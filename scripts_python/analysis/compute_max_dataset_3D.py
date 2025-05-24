"""
The idea is similar to compute_avg_std_dataset_3D but instead of computing average and standard deviation it find the max for each pixel map of each sample.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pydicom as dicom
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
# Variable to save results

max_per_file = []  # This will become an array where each element is the max for a single dcm file
max_per_slice_per_sample = [] # This will become a 2D array where each element is the max for that specific slice of that specific sample

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

        # Get order of the image
        idx_images_list = []
        for j in range(len(list_files_specific_recording)) :
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]
            idx_image = file_name.split('_')[-3]
            idx_images_list.append(int(idx_image) - 1)
        
        tmp_max_per_slice = []
        for j in range(len(list_files_specific_recording)) :
            idx = idx_images_list[j]
            file_name = list_files_specific_recording[j].split('/')[-1].split('.')[0]

            # Load the data
            data = dicom.dcmread(list_files_specific_recording[j])

            # Get and save max value
            tmp_max_per_slice.append(data.pixel_array.max())
            max_per_file.append(data.pixel_array.max())
        
        # Save max_per_slice
        max_per_slice_per_sample.append(tmp_max_per_slice)
        
        # Add the folder to the list of converted folders
        list_converted_folders.append(path_specific_recording)

# Convert to numpy array
max_per_file = np.asarray(max_per_file)
max_per_slice_per_sample = np.asarray(max_per_slice_per_sample) # Bug caused by some elements with different size

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create plot
bins = 900

fig, ax = plt.subplots(1, 1, figsize = (12, 8))
ax.hist(max_per_file, bins = bins)
ax.set_title(f"Max value for each file ({dataset_name})")
ax.set_xlabel("Max value")
ax.set_ylabel("Number of files")
# ax.set_xlim(0, 8000)
ax.grid(True)

fig.tight_layout()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

tmp_list = []
for el in max_per_slice_per_sample :
    if len(el) != 35 : print(el)
    else : tmp_list.append(el)
tmp_list = np.asarray(tmp_list)

bins = 200
fig, axs = plt.subplots(6, 6, figsize = (20, 15))
axs_list = axs.flatten()
for i in range(tmp_list.shape[1]) :
    ax = axs_list[i]
    ax.hist(tmp_list[:, i].flatten(), bins = bins)
    ax.grid(True)
    # Values for z = 39 slice = 5
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 80)
fig.suptitle(f"Max value for each slice ({dataset_name})")
fig.tight_layout()
plt.show()
