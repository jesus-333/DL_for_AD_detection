# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os 
import numpy as np

from PIL import Image as im 
import cv2 as cv

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_all_files_from_path(path_to_explore : str) :
    """
    Given a path to explore, return a list with all files in the folder and subfolders
    """

    file_path_list = []
    for path, subdirs, files in os.walk(path_to_explore):
        for name in files:
            file_path = os.path.join(path, name)
            file_path_list.append(file_path)

    return file_path_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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
