"""
Similar to 2D version but compute the mean and std for each slice of 3D dataset (see the class MRI_3D_dataset for more info)

Links with some details/discussion about it : 
https://datascience.stackexchange.com/questions/77084/how-imagenet-mean-and-std-derived
https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt

from src.dataset import support_dataset, support_dataset_ADNI, dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = 'ADNI_axial_PD_z_44_slice_4'
path_to_data = f'./data/{dataset_name}_png_V4_2/'

n_elements_to_use = -1
load_data_in_memory = False
load_data_type = 0

# preprocess_functions  = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((256, 256)),
#     torchvision.transforms.CenterCrop(224),
# ])

preprocess_functions  = torchvision.transforms.Compose([
    torchvision.transforms.Resize((176, 176)),
])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_final_mean_and_std_from_list(mean_list, std_list) :
    # Compute the mean and std for the dataset (for each channel separately)
    mean_dataset = torch.stack(mean_list).mean(dim = 0)
    std_dataset = torch.stack(std_list).mean(dim = 0)

    # print(f'Mean dataset : {mean_dataset}')
    # print(f'Std dataset  : {std_dataset}')

    return mean_dataset, std_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get folder for each sample
folders_paths_dict = support_dataset.get_all_files_from_path_divided_per_folder(path_to_data, filetype_filter = 'png')

# Get depth map order for each folder
depth_map_order_dict = support_dataset_ADNI.get_depth_map_order_all_dataset(folders_paths_dict)

folder_to_labels_dict_int = {}
for folder in folders_paths_dict : folder_to_labels_dict_int[folder] = 0

idx_to_use = np.arange(len(folders_paths_dict))
if n_elements_to_use == -1 :
    idx_to_use = np.arange(len(folders_paths_dict))
elif n_elements_to_use > len(folders_paths_dict) :
    print(f'Warning: n_elements_to_use ({n_elements_to_use}) is greater than the number of elements in the dataset ({len(folders_paths_dict)}).')
    print(f'Setting n_elements_to_use = {len(folders_paths_dict)}')
    n_elements_to_use = len(folders_paths_dict)
    idx_to_use = np.arange(len(folders_paths_dict))
else :
    idx_to_use = np.random.choice(len(folders_paths_dict), n_elements_to_use, replace = False)

# Random permutation of the index
# Note that is useful only if you print the average mean and std every n elements to check how fast (or if) they are converging
# In this way you are sure that each time you repeat the script the order of sample is different and you could rule out that the convergence is due to the order of the samples
np.random.shuffle(idx_to_use)

mean_list = []
std_list = []
n_elements_to_print = int(len(idx_to_use) / 20)

tmp_mean_list = []
tmp_std_list = []

load_data_in_memory = load_data_in_memory
load_data_type = load_data_type
MRI_all_dataset = dataset.MRI_3D_dataset(folders_paths_dict, depth_map_order_dict, folder_to_labels_dict_int, load_data_in_memory = False, preprocess_functions = preprocess_functions)

# Compute the mean and std for each image
for i in range(len(idx_to_use)) :
    if i % n_elements_to_print == 0 and i != 0 :
        print(f'Processing {i} / {len(idx_to_use)} ({round(i / len(idx_to_use) * 100, 2)}%)')
        mean, std = compute_final_mean_and_std_from_list(mean_list, std_list)

        tmp_mean_list.append(mean)
        tmp_std_list.append(std)

        print("Mean : ", round(float(mean[0]), 6))
        print("Std  : ", round(float(std[0]), 6), "\n")

    # Load image
    # Remember that the single images are loaded with torch.io.read_image. The function load the image as tensor in uint8, i.e. with range  between 0 and 255.
    # After that, inside the method in the class that read the single images, there's a division by 255 to normalize everything between 0 and 1.
    sample, _ = MRI_all_dataset[idx_to_use[i]]

    mean_list.append(sample.mean(dim = (1, 2)))
    std_list.append(sample.std(dim = (1, 2)))

final_mean, final_std = compute_final_mean_and_std_from_list(mean_list, std_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot convergence of the mean and std

x = (np.arange(len(tmp_mean_list)) + 1) * n_elements_to_print

fig, ax = plt.subplots(2, 1, figsize = (10, 10))

ax[0].plot(x, tmp_mean_list, label = 'Mean')
ax[0].set_title('Convergence of the mean')
ax[0].set_xlabel('Number of samples')
ax[0].set_ylabel('Mean value')
ax[0].grid(True)

ax[1].plot(x, tmp_std_list, label = 'Std')
ax[1].set_title('Convergence of the std')
ax[1].set_xlabel('Number of samples')
ax[1].set_ylabel('Std value')
ax[1].grid(True)

fig.tight_layout()
plt.show()

