"""
Due to the increase in size of the support_dataset.py file, I decided to split the code in different files.
This file contains the functions to get the ADNI dataset.
Note that some of the functions are specific for my folder structure.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pandas as pd
import torch
import torchvision

from . import dataset, support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_dataset(list_of_path_to_data : list, n_samples : int = -1, merge_AD_class : int = 0, print_var : bool = True, seed : int = None) :
    """
    This function is similar to get_kaggle_AD_dataset but for the ADNI dataset. It basically works as support_dataset_kaggle.get_dataset().
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

def get_dataset_V2(dataset_config : dict, percentage_split_train_val : float = 1, idx_to_use = None, seed : int = -1) :
    """
    This function is used to create the dataset when the data is stored in a single tensor file.
    The function creates two datasets: one for training and one for validation, with the data split according to the percentage_split_train_val parameter.

    TODO (?) : add the possibility to pass specific preprocessing functions to the dataset.

    Parameters
    ----------
    dataset_config : dict
        Dictionary with the configuration of the dataset. It must contain the following keys:
            - 'name_tensor_file': name of the tensor file with the data.
            - 'path_data': path to the folder with the data.
            - 'merge_AD_class': int that represent how to merge the AD class. See merge_AD_class_function for more details.
            - 'use_normalization': bool that indicates if the data should be normalized or not.
            - 'apply_rescale': bool that indicates if the data should be rescaled or not. This option is added because to reduce the size of the tensor file, in some case I saved the data as uint16. The rescale convert the data to float in the range [0, 1].
            - 'rescale_factor': Value that indicates the factor to rescale the data.
    percentage_split_train_val : float, optional
        Percentage of the data to use for training. The default is 1, which means that all the data is used for training and no validation set is created.
        If you want to create a validation set, you can set this value to a number between 0 and 1. The split is done through the function get_idx_to_split_data_V3.
        If you do not want to create a validation set, you can set this value to 1. By default, it is set to 1.
    idx_to_use : list, optional
        List with the indices of the samples to use from the dataset. If None, all the samples are used. The default is None.
        This is useful to use only a subset of the dataset, e.g. for debugging purposes or to create an imbalanced dataset.
    seed : int, optional
        Seed for the random number generator. The default is -1, which means that a random seed is sampled from the range [0, 2**32 - 1].
        If you want to use a specific seed, you can set this value to a positive integer. If you set it to 0 or a negative integer, a new random seed will be sampled.
        The seed is used to split the data in train/validation sets through the function get_idx_to_split_data_V3.
    """
    
    # Check seed and sample a new one if the value is not valid
    if seed <= 0 :
        seed = np.random.randint(0, 2**32 - 1)
        print(f"Invalid seed value. Sample new random seed {seed}")

    if percentage_split_train_val <= 0 or percentage_split_train_val > 1 :
        raise ValueError(f"Invalid value for percentage_split_train_val. It must be in the range (0, 1). Current value: {percentage_split_train_val}")

    # Get dataset info
    dataset_tensor_file_name = dataset_config['name_tensor_file']
    path_to_data = dataset_config['path_data']
    dataset_info = pd.read_csv(f'{path_to_data}dataset_info.csv')
    labels_int = dataset_info['labels_int'].to_numpy()
    labels_str = dataset_info['labels_str'].to_numpy()

    # (OPTIONAL) Merge AD classes
    labels_int = merge_AD_class_function(labels_int, labels_str, dataset_config['merge_AD_class'])

    # Get data and labels for the specific clients
    data = torch.load(f'{path_to_data}{dataset_tensor_file_name}', mmap = True)

    if idx_to_use is not None :
        data_to_use = data[idx_to_use]
        labels_int_to_use = labels_int[idx_to_use]
    else :
        data_to_use = data[:]
        labels_int_to_use = labels_int[:]
    
    # Get idx to split the  data in train/validation/test
    idx_list = support_dataset.get_idx_to_split_data_V3(labels_int_to_use, [percentage_split_train_val, 1 - percentage_split_train_val], seed)
    idx_train, idx_validation = idx_list
    
    # (OPTIONAL) Create function to normalize the data
    if dataset_config['use_normalization'] :
        # Load precomputed dataset mean and std
        # Note that to normalize the data I still used the global mean/std
        # TODO add an option to use local mean/std
        mean_dataset = torch.load(f'{path_to_data}dataset_mean.pt')
        std_dataset  = torch.load(f'{path_to_data}dataset_std.pt')

        # Create normalization function
        preprocess_functions  = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = mean_dataset, std = std_dataset)])
    else :
        preprocess_functions = None

    # Split data in train/validation/test
    if dataset_config['apply_rescale'] :
        MRI_train_dataset = dataset.MRI_dataset(data_to_use[idx_train] / dataset_config['rescale_factor'], labels_int_to_use[idx_train], preprocess_functions = preprocess_functions)

        if percentage_split_train_val < 1 :
            MRI_validation_dataset = dataset.MRI_dataset(data_to_use[idx_validation] / dataset_config['rescale_factor'], labels_int_to_use[idx_validation], preprocess_functions = preprocess_functions)
        else :
            MRI_validation_dataset = None
    else :
        MRI_train_dataset = dataset.MRI_dataset(data_to_use[idx_train], labels_int_to_use[idx_train], preprocess_functions = preprocess_functions)

        if percentage_split_train_val < 1 :
            MRI_validation_dataset = dataset.MRI_dataset(data_to_use[idx_validation], labels_int_to_use[idx_validation], preprocess_functions = preprocess_functions)
        else :
            MRI_validation_dataset = None

    return MRI_train_dataset, MRI_validation_dataset, seed

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

def filter_AD_data(labels_list : list, class_to_keep : list, return_boolean_index : bool = False) :
    """
    Create an index array where only the samples with labels in class_to_keep are kept.
    The labels_list is a list of labels, where each label is a string or an integer.
    The class_to_keep is the list of labels to keep in the index array.

    Parameters
    ----------
    labels_list : list
        List of labels, where each label is a string or an integer.
    class_to_keep : list
        List of labels to keep in the index array. The labels must be in the labels_list.
    return_boolean_index : bool, optional
        If True, the function will return a boolean index array where True means that the sample is kept and False means that the sample is not kept.
        If False, the function will return an index array with the indexes of the samples that are kept. The default is False.

    Returns
    -------
    index_array : np.ndarray
        If return_boolean_index is False, an array of integers with the indexes of the samples that are kept.
        If return_boolean_index is True, an array of booleans with the same length of labels_list, where True means that the sample is kept and False means that the sample is not kept. By default, return_boolean_index is False.
    """

    # Check that labels_list and class_to_keep are not empty
    if len(labels_list) == 0 or len(class_to_keep) == 0 :
        raise ValueError(f"labels_list and class_to_keep must not be empty. Current lengths : labels_list={labels_list}, class_to_keep={class_to_keep}")

    # Check if labels_list contains only strings or integers
    for label in labels_list :
        if not isinstance(label, (str, int)):
            raise ValueError(f"labels_list must contain only strings or integers. Current value: {label}")
    type_element_labels_list = str(type(labels_list[0]))

    # Check if class_to_keep is a list of strings or integers
    for label in class_to_keep :
        if not isinstance(label, (str, int)):
            raise ValueError(f"class_to_keep must contain only strings or integers. Current value: {label}")
    type_element_class_to_keep = str(type(class_to_keep[0]))

    # Check if the type of the elements in labels_list and class_to_keep is the same
    if type_element_labels_list != type_element_class_to_keep :
        raise ValueError(f"The type of the elements in labels_list and class_to_keep must be the same. Current types: labels_list={type_element_labels_list}, class_to_keep={type_element_class_to_keep}")

    # Create the index array
    index_array = []
    
    # Get only the indexes of the labels that are in class_to_keep
    for i in range(len(labels_list)) :
        if return_boolean_index :
            index_array.append(labels_list[i] in class_to_keep)
        else :
            if labels_list[i] in class_to_keep :
                index_array.append(i)

    return np.asarray(index_array, dtype = bool) if return_boolean_index else np.asarray(index_array, dtype = int)

def merge_AD_class_function(labels_int, labels_str, merge_AD_class : int) :
    """
    Merge the classes of the ADNI dataset according to the merge_AD_class parameter.
    If merge_AD_class == 0, no merge is applied.
    If merge_AD_class == 1, the AD class is merged with the MCI class. You obtain a binary classification problem with CN vs all the other classes (AD, MCI, EMCI, LMCI, SMC).
    if merge_AD_class == 2, all the different MCI classes are merged in a single class. You obtain a 4-class classification problem with CN vs AD vs MCI vs SMC.

    The function takes as input the labels_int and labels_str, which are the integer and string representation of the labels.
    Both of them are lists (or arrays) of the same length, where each element corresponds to a sample in the dataset.
    The class only returns the labels_int, which are the integer representation of the labels after the merge.
    """

    if merge_AD_class == 0 :
        pass
    if merge_AD_class == 1 :
        label_to_int = dict(
            CN    = 0,
            AD    = 1,
            MCI   = 1,
            EMCI  = 1,
            LMCI  = 1,
            SMC   = 1,
        )
        for i in range(len(labels_int)) : labels_int[i] = label_to_int[labels_str[i]]
    elif merge_AD_class == 2 :
        label_to_int = dict(
            CN    = 0,
            AD    = 1,
            MCI   = 2,
            EMCI  = 2,
            LMCI  = 2,
            SMC   = 3,
        )
        for i in range(len(labels_int)) : labels_int[i] = label_to_int[labels_str[i]]

    return labels_int
