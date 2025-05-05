"""
Due to the increase in size of the support_dataset.py file, I decided to split the code in different files. 
This file contains the functions to get the ADNI dataset.
Note that some of the functions are specific for my folder structure.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os 
import numpy as np

from . import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_dataset(list_of_path_to_data : list, n_samples : int = -1, merge_AD_class : int = 0, print_var : bool = True, seed : int = None) :
    """
    This function is similar to get_kaggle_AD_dataset but for the ADNI dataset. It basically works as support_dataset_kaggle.get_dataset(), 
    Note that also this function is written with my folder structure in mind.

    Parameters
    ----------
    list_of_path_to_data : list
        List with the path to the folders with the images of the ADNI dataset. Each element of the list is a path to a class of the dataset.
        E.g. list_of_path_to_dat = ['./data/ADNI_MRI_2D_Axial_png/AD/', './data/ADNI_MRI_2D_Axial_png/CN/', './data/ADNI_MRI_2D_Axial_png/MCI/']
    n_samples : int, optional
        Number of samples to get from each the dataset. The default is -1, which means that all samples are used.
        This can be specified to reduce the size of the dataset, due to the large size of the ADNI dataset.
        When specified the dataset is balanced, i.e. the same number of samples for each class.
    merge_AD_class : int
        An int that represent how to merge the AD class. 
        If 0 no merge will be applied
        If 1 all the AD class will be merged in a single class. I.e. the AD and CN will be merged in a single class.
        Other value will raise error. Default value is 0.
    print_var : bool, optional
        If True, print the number of samples for each class. The default is True.
    seed : int, optional
        Seed for the random number generator. The default is None, which means that the seed is not set.
        Note that this is used only if n_samples is not -1 (i.e. you samples a subset of the dataset).
    """

    if merge_AD_class == 0 :
        label_to_int = dict(
            CN = 0,
            AD = 1,
            MCI = 2,
        )
    elif merge_AD_class == 1 :
        label_to_int = dict(
            CN = 0,
            AD = 1,
            MCI = 1,
        )
    else :
        raise ValueError(f"merge_AD_class not valid. Must be 0 (no merge), 1 (merge all AD class). Current value : {merge_AD_class}")
    
    # Get the path to all files
    file_path_list = []
    for path_to_explore in list_of_path_to_data : file_path_list += support_dataset.get_all_files_from_path(path_to_explore, filetype_filter = 'png')

    # Convert to numpy array
    file_path_list = np.asarray(file_path_list)

    # Get labels (string)
    label_list_str = support_dataset.get_labels_from_path_list(file_path_list)

    # Convert labels to int
    label_list_int = [label_to_int[label] for label in label_list_str]
    label_list_int = np.asarray(label_list_int)

    if print_var :
        print("Number of samples for each class :")
        for key, value in label_to_int.items() :
            print("\t{} : {}".format(key, np.sum(np.asarray(label_list_int) == value)))
        print("Total number of samples in the original dataset: ", len(file_path_list))

    # (OPTIONAL) Sample the dataset
    if n_samples > 0 :
        if n_samples >= len(np.unique(label_list_int)) :
            return create_ADNI_partition(file_path_list, label_list_int, label_list_str, n_samples, seed)
        else :
            raise ValueError(f"The number of samples is less than the number of classes. Please increase the number of samples. The number of samples is {n_samples} and the number of classes is {len(np.unique(label_list_int))}.") 
    else :
        return file_path_list, label_list_int, label_list_str


def create_ADNI_partition(file_path_list : list, label_list_int : list, label_list_str : list, n_samples : int, seed : int = None) :
    """
    This function is used to create a partition of the ADNI dataset, i.e. a subset of the dataset with a specific number of samples.
    The function is the same code of the script create_ADNI_partition.py, it is simply encapsulated in a function.
    """
    
    # Set the seed for the random number generator
    if seed is not None : np.random.seed(seed)

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

    return file_path_list_sampled, label_list_int_sampled, label_list_str_sampled

def get_depth_map_order_single_sample_from_files_list(files_list : list) :
    """
    Given a list with the name of the files of a single sample (i.e. a single MRI scan), this function with analyze the name of the files to obtain the correct order of the depth map.
    Note that this function was written with the data/folder structure obtained from the script convert_all_subjects_ADNI_ONLY_2D_MRI_V4_2.py in mind.
    So you basically need to have a list of folder where each folder is a sample of the dataset.
    Then you can read the files from that specific folder, create a list with them and pass it to this function.
    The name of each files must be in the following format: idx_name.png, where idx is the index of the slice in the depth map and name is the name of the file (whatever name you want).
    """

    depth_map_order = np.zeros(len(files_list), dtype = int)

    for i in range(len(files_list)) :
        # Get the current file
        current_file = os.path.basename(files_list[i])

        # Get the index of the slice
        idx = int(current_file.split("_")[0])
        depth_map_order[idx] = i

    return np.array(depth_map_order)


def get_depth_map_order_all_dataset(folders_paths_dict  : list) :
    """
    Works as get_depth_map_order_single_sample_from_files_list but for the entire dataset.
    The input must be in the form of a dictionary where the key is the path to the folder and the value is a list with the files in that folder (e.g. the output of get_all_files_from_path_divided_per_folder).
    """

    depth_map_order_dict = {}

    for folder in folders_paths_dict.keys() :
        # Get the files in the folder
        files_list_current_folder = folders_paths_dict[folder]

        # Get the depth map order
        depth_map_order = get_depth_map_order_single_sample_from_files_list(files_list_current_folder)

        # Add to the dictionary
        depth_map_order_dict[folder] = depth_map_order
    
    return depth_map_order_dict

