"""
Given a folder with 3D dicom files visualize them through matplotlib.
A slide is used to move between different files
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_data = './data/ADNI PET/ADNI/002_S_4521/AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution/2012-03-06_16_16_29.0/I289812/'
# path_data = '../../data/ADNI PET/ADNI/002_S_4521/AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution/2012-03-06_16_16_29.0/I289812/'

sort_files = True # Remember to check if you switch from PET to MRI data

figsize = (10, 10) 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get the data

# Get all the files in the folder
list_of_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'dcm')

if sort_files :
    tmp_list = []
    idx_position_in_the_array = dict()
    for i in range(len(list_of_files)) :
        # This is valid for PET from the ADNI Dataset
        idx_image = int(list_of_files[i].split('/')[-1].split('_')[-3])

        # Save idx and its position in the array
        tmp_list.append(idx_image)
        idx_position_in_the_array[idx_image] = i
    
    # Sort idx
    idx_list = []
    tmp_list = np.sort(tmp_list)
    for idx in tmp_list : idx_list.append(idx_position_in_the_array[idx])
else :
    idx_list = np.arange(len(list_of_files))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot with slider

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize = figsize)

# Get the first image and plot it
image = dicom.dcmread(list_of_files[idx_list[0]]).pixel_array
ax.imshow(image, cmap = 'gray')
ax.set_title(f"Image {idx_list[0]}")

# Create a slider
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
slider = plt.Slider(ax_slider, 'N. image', 0, len(list_of_files) - 1, valinit = 0, valstep = 1)

def update(val) :
    """
    Update the image when the slider is moved.
    """
    # Get the index of the image
    idx = int(slider.val)
    
    # Get the image and plot it
    image = dicom.dcmread(list_of_files[idx_list[idx]]).pixel_array
    ax.imshow(image, cmap = 'gray')
    
    # Update the title
    ax.set_title(f"Image {idx_list[idx]}")
    
    # Redraw the figure
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Show the plot
fig.tight_layout()
plt.show()

