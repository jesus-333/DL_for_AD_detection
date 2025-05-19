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
import torch
import torchvision

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
    Note that this function was written with the data/folder structure obtained from the script convert/all_subjects_ADNI_ONLY_2D_MRI_V4_2.py in mind.
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

def get_labels_list_from_path_dict_V4_2(folders_paths_dict : dict, subj_to_label_dict : dict) :
    """
    Obtain the labels from the path to the folders.
    The input folder_to_labels_dict must be obtained with the following steps:
        1)Convert the data with the script convert_all_subjects_ADNI_ONLY_2D_MRI_V4_2.py
        2) Use the function get_all_files_from_path_divided_per_folder to obtain the dictionary with the path to the folders and the files in them.
    The input subj_to_label_dict must be a dictionary where each key is the ID of the a subject and the values are the corresponding labels.
    """

    labels_list = []

    for folder in folders_paths_dict.keys() :
        subj_id = folder.split("__")[0].split("/")[-1]
        try :
            labels_list.append(subj_to_label_dict[subj_id])
        except :
            raise ValueError("problem with subj_id : ", subj_id)

    return np.asarray(labels_list)

def get_labels_dict_from_path_dict_V4_2(folders_paths_dict : dict, subj_to_label_dict : dict) :
    """
    Similar to get_labels_dict_from_path_dict_V4_2() but instead of returning a list it return a dict, where each key is a key in the folders_paths_dict and each value is the corresponding label.
    """

    folder_to_labels_dict = dict()

    for folder in folders_paths_dict.keys() :
        subj_id = folder.split("__")[0].split("/")[-1]
        try :
            folder_to_labels_dict[folder] = subj_to_label_dict[subj_id]
        except :
            print(f"problem with subj_id : {subj_id}. The key wil not be inserted in the database")

    return folder_to_labels_dict

def get_preprocess_functions_ADNI_3D_png(input_size : int, use_normalization : bool = False, z_matrix : int = None, slice : int = None) :

    if input_size == 224 :
        raise ValueError("Invalid values for input_size")
    elif input_size == 176 :
        tmp_list = [torchvision.transforms.Resize((input_size, input_size))]

        if use_normalization :
            if z_matrix == 44 and slice == 4 :
                dataset_mean = torch.tensor([0.0694, 0.0674, 0.0668, 0.0681, 0.0694, 0.0718, 0.0736, 0.0767, 0.0792,
                                    0.0807, 0.0834, 0.0868, 0.0897, 0.0957, 0.1033, 0.1099, 0.1164, 0.1220,
                                    0.1242, 0.1264, 0.1272, 0.1260, 0.1284, 0.1319, 0.1339, 0.1360, 0.1367,
                                    0.1343, 0.1314, 0.1284, 0.1236, 0.1187, 0.1138, 0.1067, 0.0991, 0.0907,
                                    0.0805, 0.0686, 0.0569, 0.0492, 0.0490, 0.0557, 0.0795, 0.1302])

                dataset_std = torch.tensor([0.1019, 0.1015, 0.1020, 0.1056, 0.1096, 0.1151, 0.1199, 0.1260, 0.1312,
                                    0.1348, 0.1392, 0.1430, 0.1447, 0.1487, 0.1545, 0.1601, 0.1668, 0.1732,
                                    0.1761, 0.1785, 0.1789, 0.1781, 0.1825, 0.1885, 0.1938, 0.1990, 0.2013,
                                    0.1997, 0.1975, 0.1964, 0.1952, 0.1944, 0.1939, 0.1911, 0.1875, 0.1822,
                                    0.1737, 0.1581, 0.1361, 0.1164, 0.1091, 0.1069, 0.1005, 0.1000])
            else :
                raise ValueError("Invalid value for z_matrix and slice")

            tmp_list.append(torchvision.transforms.Normalize(mean = dataset_mean, std = dataset_std))

    preprocess_functions  = torchvision.transforms.Compose(tmp_list)

    return preprocess_functions
