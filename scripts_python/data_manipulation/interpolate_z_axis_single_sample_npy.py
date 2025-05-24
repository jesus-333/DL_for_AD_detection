"""
When I use/analyze the ADNI data (axial) z-axis has different dimensions (i.e. different number of slice), based on which scan I download.
Whit this script I interpolate the z-axis of two different samples to obtain the same number of slice in both samples.
The z-maps of sample 2 is interpolated to obtain the same z-dimension of sample 1
The interpolation is done with the numpy function `np.interp` (default settings)

The results is then visualized with a matplotlib with a slider to change between different slices.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom as dicom

from src.dataset import support_dataset, support_dataset_ADNI

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_dataset_1 = './data/ADNI_axial_PD_z_35_slice_5/'
path_dataset_2 = './data/ADNI_axial_PD_z_44_slice_4/'

figsize = (9, 9)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_depth_map_order_single_sample_from_files_list(files_list : list) :

    depth_map_order = np.zeros(len(files_list), dtype = int)
    idx_images_list = []
    for i in range(len(files_list)) :
        current_file = os.path.basename(files_list[i])
        idx = int(current_file.split("_")[-3])
        idx_images_list.append(idx)
    
    idx_to_use = dict()
    sorted_idx = np.sort(idx_images_list)
    for i in range(len(idx_images_list)) :
        idx = sorted_idx[i]
        idx_to_use[idx] = i

    for i in range(len(files_list)) :
        # Get the current file
        current_file = os.path.basename(files_list[i])

        # Get the index of the slice
        idx = int(current_file.split("_")[-3])
        depth_map_order[idx_to_use[idx]] = i

    return np.array(depth_map_order)

def get_conversion_factor(pixel_max_value : int) :
    """
    Get the factor to scale the pixel value between 0 and 4095
    """
    if pixel_max_value >= 0 and pixel_max_value <= 4095 :
        return 1
    elif pixel_max_value > 4095 and pixel_max_value <= 8191 :
        return 2
    elif pixel_max_value > 8191 and pixel_max_value <= 16383 :
        return 4
    elif pixel_max_value > 16383 and pixel_max_value <= 32768 :
        return 8
    elif pixel_max_value > 32768 and pixel_max_value <= 65535 :
        return 16
    else :
        raise ValueError(f"max pixel_max_value outside valid range. The value must be between 0 and 65535. The current value is {pixel_max_value}.")

def convert_list_of_dcm_files_to_3D_tensor(files_list : list, depth_map_order : list) :
    sample_3D_tensor = []

    for i in range(len(files_list)) :
        pixel_data = dicom.dcmread(files_list[depth_map_order[i]]).pixel_array
    
        # Convert in range 0-4095
        conversion_factor = get_conversion_factor(pixel_data.max())
        pixel_data = pixel_data / conversion_factor

        # Convert in range 0-1
        # pixel_data = pixel_data / 4095

        sample_3D_tensor.append(pixel_data)

    return np.asarray(sample_3D_tensor)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get all the files divided per folder
folder_paths_dict_dataset_1 = support_dataset.get_all_files_from_path_divided_per_folder(path_dataset_1, filetype_filter = 'dcm')
folder_paths_dict_dataset_2 = support_dataset.get_all_files_from_path_divided_per_folder(path_dataset_2, filetype_filter = 'dcm')

# Random sample 1 folder for each dataset
idx_1 = np.random.randint(0, len(folder_paths_dict_dataset_1))
idx_2 = np.random.randint(0, len(folder_paths_dict_dataset_2))
files_list_sample_datase_1 = folder_paths_dict_dataset_1[list(folder_paths_dict_dataset_1)[idx_1]]
files_list_sample_datase_2 = folder_paths_dict_dataset_2[list(folder_paths_dict_dataset_2)[idx_2]]

# Get depth map order for each folder
depth_map_order_dict_sample_1 = get_depth_map_order_single_sample_from_files_list(files_list_sample_datase_1)
depth_map_order_dict_sample_2 = get_depth_map_order_single_sample_from_files_list(files_list_sample_datase_2)

# Create 3D tensor for the two samples
sample_3D_tensor_1 = convert_list_of_dcm_files_to_3D_tensor(files_list_sample_datase_1, depth_map_order_dict_sample_1)
sample_3D_tensor_2 = convert_list_of_dcm_files_to_3D_tensor(files_list_sample_datase_2, depth_map_order_dict_sample_2)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Interpolate the z-axis of the two samples
sample_3D_tensor_2_interpolated = np.zeros((sample_3D_tensor_1.shape[0], sample_3D_tensor_2.shape[1], sample_3D_tensor_2.shape[2]))

original_points = np.linspace(0, 1, sample_3D_tensor_2.shape[0])
new_points = np.linspace(0, 1, sample_3D_tensor_1.shape[0])

# Iterate over high and width of the image
for i in range(sample_3D_tensor_2.shape[1]) : 
    for j in range(sample_3D_tensor_2.shape[2]) :
        # Interpolate data
        data_to_interpolate = sample_3D_tensor_2[:, i, j]
        data_interpolated = np.interp(new_points, original_points, data_to_interpolate)
    
        # Save data
        sample_3D_tensor_2_interpolated[:, i, j] = data_interpolated

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot Original image

# Create figure and slider
fig, ax = plt.subplots(1, 1, figsize = figsize)
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
slider = plt.Slider(ax_slider, 'z-axis', 0, sample_3D_tensor_2.shape[0] - 1, valinit = 0, valstep = 1)

# Plot the first image
ax.imshow(sample_3D_tensor_2[0], cmap = 'gray')
ax.set_title('Original image')

def update(val) :
    idx = int(slider.val)
    img = sample_3D_tensor_2[idx].squeeze()
    ax.imshow(img, cmap = 'gray')
    ax.set_title(f"z-axis : {idx}")
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Show the plot
fig.tight_layout()
fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot Interpolated image

# Create figure and slider
fig, ax = plt.subplots(1, 1, figsize = figsize)
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
slider = plt.Slider(ax_slider, 'z-axis', 0, sample_3D_tensor_2_interpolated.shape[0] - 1, valinit = 0, valstep = 1) 

# Plot the first image 
ax.imshow(sample_3D_tensor_2_interpolated[0], cmap = 'gray') 
ax.set_title('Interpolated image')

def update(val) : 
    idx = int(slider.val)
    img = sample_3D_tensor_2_interpolated[idx].squeeze()
    ax.imshow(img, cmap = 'gray')
    ax.set_title(f"z-axis : {idx}")
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Show the plot
fig.tight_layout()
fig.show()
