"""
Due to the increase in size of the support_dataset.py file, I decided to split the code in different files. 
This file contains the functions to get the kaggle AD dataset.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import numpy as np
import shutil

import torch
from torchvision import transforms

from . import dataset_png, support_dataset

try :
    import kagglehub
    kagglehub_installed = True
except ImportError :
    print("kagglehub not installed. The function download_and_get_kaggle_AD_dataset will not work.")
    kagglehub_installed = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_data(path_files_Moderate_Demented : str, path_files_Mild_Demented : str, path_files_Very_Mild_Demented : str, path_files_Non_Demented : str, 
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
        label_to_int = dict(
            NonDemented = 0,
            ModerateDemented = 1,
            MildDemented = 2,
            VeryMildDemented = 3
        )
    elif merge_AD_class == 1 :
        label_to_int = dict(
            NonDemented = 0,
            ModerateDemented = 1,
            MildDemented = 1,
            VeryMildDemented = 1
        )
    elif merge_AD_class == 2 :
        label_to_int = dict(
            NonDemented = 0,
            ModerateDemented = 1,
            MildDemented = 2,
            VeryMildDemented = 2
        )
    else :
        raise ValueError(f"merge_AD_class not valid. Must be 0 (no merge), 1 (merge all AD class) or 2 (merge Mild and Very Mild). Current value : {merge_AD_class}")

    # Get the paths of the files for each class
    file_path_list_1 = support_dataset.get_all_files_from_path(path_files_Moderate_Demented)
    file_path_list_2 = support_dataset.get_all_files_from_path(path_files_Mild_Demented)
    file_path_list_3 = support_dataset.get_all_files_from_path(path_files_Very_Mild_Demented)
    file_path_list_control = support_dataset.get_all_files_from_path(path_files_Non_Demented)
    if print_var : print("Paths of files readed")

    # Merge all paths and convert to nmupy array
    file_path_list = file_path_list_1 + file_path_list_2 + file_path_list_3 + file_path_list_control
    file_path_list = np.asarray(file_path_list)

    # Get labels
    label_list_str = support_dataset.get_labels_from_path_list(file_path_list)

    # Convert labels
    label_list_int = [label_to_int[label] for label in label_list_str]
    label_list_int = np.asarray(label_list_int)

    # Print the number of samples for each class
    if print_var :
        print("Number of samples for each class :")
        if merge_AD_class == 0 :
            print("\tNonDemented      : {}".format(np.sum(np.asarray(label_list_int) == 0)))
            print("\tModerateDemented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
            print("\tMildDemented     : {}".format(np.sum(np.asarray(label_list_int) == 2)))
            print("\tVeryMildDemented : {}".format(np.sum(np.asarray(label_list_int) == 3)))
        if merge_AD_class == 1 :
            print("\tControl  : {}".format(np.sum(np.asarray(label_list_int) == 0)))
            print("\tDemented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
        elif merge_AD_class == 2:
            print("\tNonDemented      : {}".format(np.sum(np.asarray(label_list_int) == 0)))
            print("\tModerateDemented : {}".format(np.sum(np.asarray(label_list_int) == 1)))
            print("\tMildDemented     : {}".format(np.sum(np.asarray(label_list_int) == 2)))
        print("Total number of samples : {}\n".format(len(label_list_int)))

    return file_path_list, label_list_int, label_list_str


def download_and_get_data(path_to_save_data : str = './data/', merge_AD_class : bool = False, print_var : bool = True) :
    """
    Download and return the Kaggle AD dataset. The dataset can be find at the following link : https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset
    The data are downloaded in the default path of kagglehub and then moved in the new path provided. If no path is provided, the data are moved in the './data/' folder.
    If the dataset is already present in the specified location, the function will not download the data again.
    """

    if not kagglehub_installed :
        raise ValueError("kagglehub not installed. The function download_and_get_kaggle_AD_dataset will not work. To install kagglehub use the following command : !pip install kagglehub")
    else :
        # Check if the data are already present in the specified location
        # TODO : Check if it works also with non default path
        if not os.path.exists(os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset')) :
            # Download the dataset and move to the new path
            path_dataset = kagglehub.dataset_download("marcopinamonti/alzheimer-mri-4-classes-dataset")
            path_dataset += '/Alzheimer_MRI_4_classes_dataset/'
            shutil.move(path_dataset, path_to_save_data)
        
        # New path of the data
        path_files_Moderate_Demented    = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/MildDemented')
        path_files_Mild_Demented        = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/ModerateDemented')
        path_files_Very_Mild_Demented   = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/VeryMildDemented')
        path_files_Non_Demented         = os.path.join(path_to_save_data, 'Alzheimer_MRI_4_classes_dataset/NonDemented')
        print(path_files_Moderate_Demented)

    return get_data(path_files_Non_Demented, path_files_Mild_Demented, path_files_Moderate_Demented, path_files_Very_Mild_Demented, merge_AD_class, print_var)

def get_dataset_with_preprocess_function_from_data(file_path_list : list, label_list_int : list, model_config : dict, dataset_config : dict) :
    # Get proprocess function
    if model_config['input_size'] == 224 :
        dataset_mean = torch.tensor([0.4233, 0.4233, 0.4233]) if not dataset_config['grey_scale_image'] else torch.tensor([0.4233])
        dataset_std  = torch.tensor([0.3179, 0.3179, 0.3179]) if not dataset_config['grey_scale_image'] else torch.tensor([0.3179])

        preprocess_functions  = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(model_config['input_size']),
            transforms.Normalize(mean = dataset_mean, std = dataset_std),
        ])
    elif model_config['input_size'] == 176 :
        # This values are precomputed with the script compute_avg_std_dataset.py (using the Resize(176)  before computation)
        dataset_mean = torch.tensor([0.2816, 0.2816, 0.2816]) if not dataset_config['grey_scale_image'] else torch.tensor([0.2816])
        dataset_std  = torch.tensor([0.3259, 0.3259, 0.3259]) if not dataset_config['grey_scale_image'] else torch.tensor([0.3259])

        preprocess_functions  = transforms.Compose([
            transforms.Resize((model_config['input_size'], model_config['input_size'])),
            transforms.Normalize(mean = dataset_mean, std = dataset_std),
        ])
    else :
        raise ValueError("Input size not supported. Use 224 or 176")

    load_data_in_memory = dataset_config['load_data_in_memory']
    test_dataset = dataset_png.MRI_2D_dataset(file_path_list, label_list_int, load_data_in_memory = load_data_in_memory, 
                                              preprocess_functions = preprocess_functions, grey_scale_image = dataset_config['grey_scale_image']
                                              )

    return test_dataset




