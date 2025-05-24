"""
Since the ADNI dataset is particularly large, this script can be used to sample a subset of n elements from the dataset.
The elements are selected randomly. The proportion of the classes is preserved (if not specified otherwise)(NOT YET IMPLEMENTED).
Note that the script is created to be used with my folder structure.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import shutil

from src.dataset import support_dataset_ADNI

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = 'ADNI_axial_PD_T2_TSE_png'
path_to_data = f"./data/{dataset_name}/"

n_samples = 20000
seed = 333

print_var = True
path_to_save = f'./data/{dataset_name}_sampled_{n_samples}/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the folder if it does not exist
if not os.path.exists(path_to_save) : 
    os.makedirs(path_to_save)
else :
    raise ValueError(f"Path to save {path_to_save} already exists. Please delete it (or rename it) before running the script again. This will avoid bug")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Sample data

if seed is not None :
    np.random.seed(seed)

# Get path to files and labels.
list_of_path_to_data = [path_to_data + 'AD/', path_to_data + 'CN/', path_to_data + 'MCI/']
file_path_list, label_list_int, label_list_str = support_dataset_ADNI.get_dataset(list_of_path_to_data, print_var = print_var)
if print_var : print("Total number of samples in the original dataset: ", len(file_path_list))

# Get unique labels
label_list_str = np.array(label_list_str)
unique_labels = np.unique(label_list_str)

# Variable to store the sampled data
file_path_list_sampled = []
label_list_int_sampled = []
label_list_str_sampled = []

n_samples_per_label = n_samples // len(unique_labels)

# Sample data
for i in range(len(unique_labels)) :
    label = unique_labels[i]
    idx_label = np.where(label_list_str == label)[0]
    
    # Check if there are enough samples for the label
    if len(idx_label) < n_samples_per_label :
        raise ValueError(f"Not enough samples for label {label}. Only {len(idx_label)} samples available.")
    
    # Randomly select n_samples from the list of files
    idx_label = np.random.choice(idx_label, n_samples_per_label, replace = False)

    # Append the selected samples to the sampled lists
    file_path_list_sampled += list(file_path_list[idx_label])
    label_list_int_sampled += list(label_list_int[idx_label])
    label_list_str_sampled += list(label_list_str[idx_label])

# Check if the number of samples is correct (should be n_samples but if the number of samples is not divisible by the number of classes, it will be less)
if len(file_path_list_sampled) != n_samples :
    print(f"WARNING: The number of samples is not equal to {n_samples}. It is {len(file_path_list_sampled)}. This is due to the fact that the number of samples is not divisible by the number of classes.")

    n_extra_samples = n_samples - len(file_path_list_sampled)
    print(f"{n_extra_samples} samples will be added randomly.")

    # Randomly select n_extra_samples from the list of files
    # Yeah I know that in this way there is a probability that the same sample is selected twice, but for now I ignore it. In the future I will eventually fix it.
    # Also if the number of samples is much smaller than the number of elements in the dataset, this probability is very, very low.
    idx_extra_samples = np.random.choice(len(file_path_list), n_extra_samples, replace = False)

    # Append the selected samples to the sampled lists
    file_path_list_sampled += list(file_path_list[idx_extra_samples])
    label_list_int_sampled += list(label_list_int[idx_extra_samples])
    label_list_str_sampled += list(label_list_str[idx_extra_samples])
    
# Convert to numpy arrays
file_path_list_sampled = np.array(file_path_list_sampled)
label_list_int_sampled = np.array(label_list_int_sampled)
label_list_str_sampled = np.array(label_list_str_sampled)

# Print the number of samples per label
if print_var :
    print("Number of samples per label :")
    for i in range(len(unique_labels)) :
        label = unique_labels[i]
        idx_label = np.where(label_list_str_sampled == label)[0]
        print(f"\t{label} : {len(idx_label)}")
    print("Total number of samples in the sampled dataset: ", len(file_path_list_sampled))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save data

# Save the sampled data
for i in range(len(unique_labels)) :
    label = unique_labels[i]
    os.makedirs(path_to_save + label + '/', exist_ok = True)

    idx_label = np.where(label_list_str_sampled == label)[0]

    for j in range(len(idx_label)) :
        file_path = file_path_list_sampled[idx_label[j]]
        
        file_name = os.path.basename(file_path).split('.')[0]
        file_extension = os.path.basename(file_path).split('.')[1]
        new_file_name_with_extension = f'{file_name}_{label}.{file_extension}'

        shutil.copy(file_path, path_to_save + label + '/' + os.path.basename(new_file_name_with_extension))
