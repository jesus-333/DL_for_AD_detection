"""
Test the class  MRI_3D_dataset

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import torch
from torchvision import transforms

from src.dataset import dataset, support_dataset, support_dataset_ADNI

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = "ADNI_axial_PD_z_44_slice_4_png_V4_2"
path_all_data = f"./data/{dataset_name}/"

plot_sample = False

input_size = 176
tmp_list = [transforms.Resize((176, 176))]
preprocess_functions  = transforms.Compose(tmp_list)

try_load_dataset_in_memory = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get all the files divided per folder
folders_paths_dict = support_dataset.get_all_files_from_path_divided_per_folder(path_all_data, filetype_filter = 'png')

# Get depth map order for each folder
depth_map_order_dict = support_dataset_ADNI.get_depth_map_order_all_dataset(folders_paths_dict)

# Random sample folder
folder_name = list(depth_map_order_dict.keys())[np.random.randint(0, len(depth_map_order_dict))]

# Print info to check if the depth map order is correct
print("" + "-" * 50)
# print(f"Check depth map order for folder {folder_name}\n(Print max of 5 files for clarity)\n")
# for i in range(min(5, len(depth_map_order_dict[folder_name]))) :
#     print(f"Depth map order : {depth_map_order_dict[folder_name][i]}")
#     print(f"File path : {folders_paths_dict[folder_name][i]}\n")

# Print the first n_elment_to_print files according to the depth map order (or the first n files if there are less than n_elment_to_print)
n_elment_to_print = 11
print(f"First {n_elment_to_print} files according to the depth map order (or the first n files if there are less than {n_elment_to_print}) :\n")
for i in range(min(n_elment_to_print, len(depth_map_order_dict[folder_name]))) :
    idx_image = depth_map_order_dict[folder_name][i]
    print(f"File path : {folders_paths_dict[folder_name][idx_image]}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create dataset

# Create fake labels
labels = {}
for folder in folders_paths_dict : labels[folder] = 1

# Create dataset
example_dataset = dataset.MRI_3D_dataset(folders_paths_dict, depth_map_order_dict, labels, preprocess_functions = preprocess_functions)

# Get a sample
idx = np.random.randint(0, len(folders_paths_dict))
sample, _ = example_dataset[idx]
print("\nSingle sample shape : ", sample.shape)

dataloader = torch.utils.data.DataLoader(example_dataset, batch_size = 4, shuffle = True)

# Get a batch
print("\nBatch sample shape : ")
for i, (images, labels) in enumerate(dataloader) :
    print(images.shape, labels.shape)
    break


# Check a single sample
if plot_sample :
    example_dataset.check_single_sample_V1(idx)
    example_dataset.check_single_sample_V2(idx)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if try_load_dataset_in_memory :
    example_dataset = dataset.MRI_3D_dataset(folders_paths_dict, depth_map_order_dict, labels, 
                                             preprocess_functions = preprocess_functions, load_data_in_memory = try_load_dataset_in_memory
                                             )
    
    print("Data loaded shape :")
    print(example_dataset.data_loaded.shape)
