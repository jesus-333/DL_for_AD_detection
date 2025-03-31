"""
Support functions for dataset management.
Note that some of the functions are specific for my folder structure.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os 
import numpy as np
import cv2 as cv
import shutil

import torch
from torchvision import transforms

try :
    import kagglehub
    kagglehub_installed = True
except ImportError :
    print("kagglehub not installed. The function download_and_get_kaggle_AD_dataset will not work.")
    kagglehub_installed = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_idx_to_split_data_V1(n_elements : int, percentage_split : float, seed : int = None):
    """
    Given a number of elements (n_elements) create an array with number from 0 to n_elements - 1 and split it (randomly) in two lists.
    The size of the two list is determined by the percentage_split parameter. The first list will be have size x = int(percentage_split * n_elements) while the second will have size y = n_elements - x
    The procedure can be "deterministic" if the seed parameter is passed to the function.
    """
    
    # Check input parameter
    if n_elements <= 1 : raise ValueError("n_elements must be greater than 1. Current value is {}".format(n_elements))
    if percentage_split <= 0 or percentage_split >= 1 : raise ValueError("percentage_split must be between 0 and 1. Current value is {}".format(percentage_split))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Create idx vector
    idx = np.random.permutation(n_elements)
    size_1 = int(n_elements * percentage_split) 
    
    return idx[0:size_1], idx[size_1:]

def random_split_array(input_aray,  percentage_split : float, seed : None) :
    """
    Given an input array, split it in two arrays with a specific percentage.
    The seed parameter can be used to make the split deterministic.
    """

    # Check input parameter
    if percentage_split <= 0 or percentage_split >= 1 : raise ValueError("percentage_split must be between 0 and 1. Current value is {}".format(percentage_split))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Split the array
    idx_1, idx_2 = get_idx_to_split_data_V1(len(input_aray), percentage_split)

    return input_aray[idx_1], input_aray[idx_2]

def get_idx_to_split_data_V2(n_elements : int, percentage_split_list : list, seed : int = None):
    """
    Given a number of elements (n_elements) create an array with number from 0 to n_elements - 1 and split it (randomly) in n lists.
    Each of the n list will have a percentage of elements determined by the percentage_split_list parameter. The sum of the elements in percentage_split_list must be equal to 1.

    E.g. n_elements = 100, percentage_split_list = [0.6, 0.3, 0.1]. The first list will have 60 elements, the second 30 and the third 10.
    The procedure can be "deterministic" if the seed parameter is passed to the function.

    Parameters
    ----------
    n_elements : int
        Number of elements to split
    percentage_split_list : list
        List with the percentage of elements for each split
    seed : int
        Seed for reproducibility. Default is None.
    """
    
    # Check input parameter
    # Note that check for 0.9999999999999999 is due to the float precision
    if n_elements <= 1 : raise ValueError("n_elements must be greater than 1. Current value is {}".format(n_elements))
    if np.sum(percentage_split_list) != 1 and np.sum(percentage_split_list) != 0.9999999999999999 : raise ValueError("The sum of the elements in percentage_split_list must be equal to 1. Current sum is {}".format(np.sum(percentage_split_list)))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Create idx vector
    idx_to_split = np.arange(n_elements).astype(int)

    # Create splits with the idx
    idx_list = []
    for i in range(len(percentage_split_list) - 1) :
        percentage = percentage_split_list[i]
        
        if i == 0 :
            actual_percentage = percentage
        else :
            size_split = int(percentage * n_elements)
            actual_percentage = size_split / len(idx_to_split)

        idx_to_save, idx_to_split = random_split_array(idx_to_split, actual_percentage, seed)
        idx_list.append(idx_to_save)

    # The last split is the remaining idx
    idx_list.append(idx_to_split)

    return idx_list

def get_idx_to_split_data_V3(labels_list : list, percentage_split_list : list, seed : int = None):
    """
    Given a list of labels (labels_list) create an array with number from 0 to len(labels_list) - 1 and split it in n lists.
    Each of the n list will have a percentage of elements determined by the percentage_split_list parameter. The sum of the elements in percentage_split_list must be equal to 1.
    The proportion of the labels in the splits is preserved, i.e. if the label 'A' is 10% of the dataset, in each split there will be 10% elements with label 'A'.
    Of course this is possibile only if there are enough elements for each label in the dataset.
    
    Parameters
    ----------
    labels_list : list
        List with the labels
    percentage_split_list : list
        List with the percentage of elements for each split
    seed : int
        Seed for reproducibility. Default is None.
    """

    # Check input parameter
    # Note that check for 0.9999999999999999 is due to the float precision
    if np.sum(percentage_split_list) != 1 and np.sum(percentage_split_list) != 0.9999999999999999 : raise ValueError("The sum of the elements in percentage_split_list must be equal to 1. Current sum is {}".format(np.sum(percentage_split_list)))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Get the unique labels
    unique_labels = np.unique(labels_list)

    # Convert the labels_list to numpy array
    labels_list = np.asarray(labels_list)
    
    # Create list with list of indices
    idx_list = []
    for i in range(len(percentage_split_list)) : idx_list.append([])

    for unique_label in unique_labels :
        # Get the idx for the unique label
        idx_for_current_label = np.where(labels_list == unique_label)[0]

        # Get split for current indices
        tmp_split_for_current_label = get_idx_to_split_data_V2(len(idx_for_current_label), percentage_split_list, seed)

        for i in range(len(percentage_split_list)) :
            idx_list[i] = idx_list[i] + list(idx_for_current_label[tmp_split_for_current_label[i]]) 
    
    return idx_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_all_files_from_path(path_to_explore : str, filetype_filter : str = None) :
    """
    Given a path to explore, return a list with all files in the folder and subfolders
    If the argument filetype_filter  is passed, only the files with the specified extension are returned
    """

    file_path_list = []
    for path, subdirs, files in os.walk(path_to_explore):
        for name in files:
            file_path = os.path.join(path, name)
            
            if filetype_filter is not None :
                if file_path.endswith(filetype_filter) : file_path_list.append(file_path)
            else :
                file_path_list.append(file_path)

    return file_path_list

def convert_label_from_str_to_int(labels_str_list : list, convert_MCI_to_AD : bool = False) :
    tmp_array = np.asarray(labels_str_list)

    labels_int_list = np.zeros(len(tmp_array))
    labels_int_list[tmp_array == 'CN'] = 0
    labels_int_list[tmp_array == 'AD'] = 1

    if convert_MCI_to_AD : labels_int_list[tmp_array == 'MCI'] = 1
    else : labels_int_list[tmp_array == 'MCI'] = 2

    return labels_int_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function specific for me code repository

def convert_3D_matrix_to_image(data, path_to_save : str, file_name : str, axis_to_save : int, min_idx_to_save : int = 0, max_idx_to_save : int = -1) :
    """
    Convert a 3D matrix to a set of images along a specific axis
    """

    os.makedirs(path_to_save, exist_ok = True)

    if max_idx_to_save == -1 : max_idx_to_save = data.shape[axis_to_save]

    for i in range(data.shape[axis_to_save]) :
        # Skip the slices with idx lower than min_idx_to_save or higher than max_idx_to_save
        if i < min_idx_to_save or i > max_idx_to_save : continue

        if axis_to_save == 0 :
            data_to_save = data[i, :, :]
        elif axis_to_save == 1 :
            data_to_save = data[:, i, :]
        elif axis_to_save == 2 : 
            data_to_save = data[:, :, i]
        else : 
            raise ValueError("Axis to save not valid. Must be 0, 1 or 2. Current value : {}".format(axis_to_save))
        
        final_path = path_to_save + '{}_{}.png'.format(file_name, i) 
        cv.imwrite(final_path, data_to_save) 


def get_labels_from_path_list(file_paths_list : str) :
    """
    This function is specific for me, due to how I saved the data. Each class of the dataset (AD, CN, MCI) has its own subfolder. 
    So from the list with all the paths, for each path, I can extract the subfolder and use it as a label for the corresponding path.
    Note that this function is specific for my folder structure.
    """
    
    labels_list = []
    for file_path in file_paths_list :
        label = file_path.split('/')[3]
        labels_list.append(label)

    return labels_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Kaggle AD dataset specific functions

def get_kaggle_AD_dataset(path_files_Moderate_Demented : str, path_files_Mild_Demented : str, path_files_Very_Mild_Demented : str, path_files_Non_Demented : str, 
                          merge_AD_class : int = 0, print_var : bool = True) :
    """
    Given the path to the folders with the images of the AD dataset, return the list with the paths of the files and the labels.
    The dataset can be find at the following link : https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset

    Parameters
    ----------
    path_files_Non_Demented : str
        Path to the folder with the images of the NonDemented class
    path_files_Mild_Demented : str
        Path to the folder with the images of the MildDemented class
    path_files_Moderate_Demented : str
        Path to the folder with the images of the ModerateDemented class
    path_files_Very_Mild_Demented : str
        Path to the folder with the images of the VeryMildDemented class
    merge_AD_class : int
        An int that represent how to merge the AD class. 
        If 0 no merge will be applied
        If 1 all the AD class will be merged in a single class. I.e. the MildDemented, ModerateDemented and VeryMildDemented will be merged in a single class.
        If 2 only the MildDemented and VeryMildDemented will be merged in a single class.
        Default is 0.
    print_var : bool
        If True, print some information about the dataset. Default is True.

    Returns
    -------
    file_path_list : np.ndarray
        List with the paths of the files. I.e. each element of the list is a path to an image of the dataset.
    label_list_int : np.ndarray
        List with the labels. Each element of the list is an integer representing the class of the corresponding image. The length of the list is the same as the file_path_list.
        The first element of the list is the label for the first element of the file_path_list and so on.
    label_list_str : list
        The same as label_list_int but with the labels in string format.
    """

    if merge_AD_class == 0 :
        pass
    elif merge_AD_class == 1 :
        label_to_int = dict(
            NonDemented = 0,
            ModerateDemented = 1,
            MildDemented = 2,
            VeryMildDemented = 3
        )
        label_to_int['MildDemented'] = label_to_int['VeryMildDemented'] = label_to_int['ModerateDemented'] = 1
    elif merge_AD_class == 2 :
        label_to_int = dict(
            NonDemented = 0,
            ModerateDemented = 1,
            MildDemented = 2,
            VeryMildDemented = 2
        )
        label_to_int['MildDemented'] = label_to_int['VeryMildDemented'] = 1
    else :
        raise ValueError(f"merge_AD_class not valid. Must be 0 (no merge), 1 (merge all AD class) or 2 (merge Mild and Very Mild). Current value : {merge_AD_class}")

    # Get the paths of the files for each class
    file_path_list_1 = get_all_files_from_path(path_files_Moderate_Demented)
    file_path_list_2 = get_all_files_from_path(path_files_Mild_Demented)
    file_path_list_3 = get_all_files_from_path(path_files_Very_Mild_Demented)
    file_path_list_control = get_all_files_from_path(path_files_Non_Demented)
    if print_var : print("Paths of files readed")

    # Merge all paths and convert to nmupy array
    file_path_list = file_path_list_1 + file_path_list_2 + file_path_list_3 + file_path_list_control
    file_path_list = np.asarray(file_path_list)

    # Get labels
    label_list_str = get_labels_from_path_list(file_path_list)

    # Convert labels
    label_list_int = [label_to_int[label] for label in label_list_str]
    label_list_int = np.asarray(label_list_int)

    # Print the number of samples for each class
    if print_var :
        print("Number of samples for each class :")
        if merge_AD_class :
            print("Control  : {}".format(np.sum(np.asarray(label_list_int) == 0)))
            print("Demented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
        else :
            print("NonDemented      : {}".format(np.sum(np.asarray(label_list_int) == 0)))
            print("ModerateDemented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
            print("MildDemented     : {}".format(np.sum(np.asarray(label_list_int) == 2)))
            print("VeryMildDemented : {}".format(np.sum(np.asarray(label_list_int) == 3)))
        print("Total number of samples : {}\n".format(len(label_list_int)))

    return file_path_list, label_list_int, label_list_str


def download_and_get_kaggle_AD_dataset(path_to_save_data : str = './data/', merge_AD_class : bool = False, print_var : bool = True) :
    """
    Download and return the Kaggle AD dataset. The dataset can be find at the following link : https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset
    The data are downloaded in the default path of kagglehub and then moved in the new path provided. If no path is provided, the data are moved in the './data/' folder.
    """

    if not kagglehub_installed :
        raise ValueError("kagglehub not installed. The function download_and_get_kaggle_AD_dataset will not work. To install kagglehub use the following command : !pip install kagglehub")
    else :
        # Download the dataset and move to the new path
        path_dataset = kagglehub.dataset_download("marcopinamonti/alzheimer-mri-4-classes-dataset")
        shutil.move(path_dataset, path_to_save_data)
        
        # New path of the data
        path_files_Moderate_Demented    = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/MildDemented')
        path_files_Mild_Demented        = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/ModerateDemented')
        path_files_Very_Mild_Demented   = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/VeryMildDemented')
        path_files_Non_Demented         = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/NonDemented')
        print(path_files_Moderate_Demented    )

    return get_kaggle_AD_dataset(path_files_Non_Demented, path_files_Mild_Demented, path_files_Moderate_Demented, path_files_Very_Mild_Demented, merge_AD_class, print_var)

def get_preprocess_function(dataset_name : str, input_size : int, grey_scale_image : bool) :
    if dataset_name == 'kaggle' :

        # This values are precomputed with the script compute_avg_std_dataset.py (withoug using the CenterCrop and Resize)
        # dataset_mean = torch.tensor([0.2816, 0.2816, 0.2816])
        # dataset_std  = torch.tensor([0.3269, 0.3269, 0.3269])

        if input_size == 224 :
            dataset_mean = torch.tensor([0.4233, 0.4233, 0.4233]) if not grey_scale_image else torch.tensor([0.4233])
            dataset_std  = torch.tensor([0.3179, 0.3179, 0.3179]) if not grey_scale_image else torch.tensor([0.3179])

            preprocess_functions  = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.Normalize(mean = dataset_mean, std = dataset_std),
            ])
        elif input_size == 176 :
            # This values are precomputed with the script compute_avg_std_dataset.py (using the Resize(176)  before computation)
            dataset_mean = torch.tensor([0.2816, 0.2816, 0.2816]) if not grey_scale_image else torch.tensor([0.2816])
            dataset_std  = torch.tensor([0.3259, 0.3259, 0.3259]) if not grey_scale_image else torch.tensor([0.3259])

            preprocess_functions  = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.Normalize(mean = dataset_mean, std = dataset_std),
            ])
        else :
            raise ValueError("Input size not supported for kaggle. Use 224 or 176")
    else :
        raise ValueError("dataset_name not valid")
    
    return preprocess_functions, dataset_mean, dataset_std

