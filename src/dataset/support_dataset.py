# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os 
import numpy as np

from PIL import Image as im 
import cv2 as cv

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
    This function is specific for me, due to how I saved the data.
    Each class of the dataset (AD, CN, MCI) has its own subfolder. 
    So from the list with all the paths, for each path, I can extract the subfolder and use it as a label for the corresponding path
    """
    
    labels_list = []
    for file_path in file_paths_list :
        label = file_path.split('/')[3]
        labels_list.append(label)

    return labels_list
