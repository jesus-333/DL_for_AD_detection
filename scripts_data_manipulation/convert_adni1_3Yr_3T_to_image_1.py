"""
This script convert the data of the dataset ADNI1_3Yr_3T to images.
The original data are saved in nii files. This script read the nii files and convert them to png file along a specific axis
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import nibabel as nib

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_save = './data/ADNI_1_3YR_3T'
min_idx_to_save = 60
max_idx_to_save = -1
filter_preprocess = True

# The nii files contains 3D matrices. If I fix 1 axis and move along it I will obtain 2D slices that can be saved as images.
axis_to_save = 0 # 0 --> View from above, 1 --> View from front, 2 --> View from side
axis_int_to_label = { 0 : 'above', 1 : 'fron', 2 : 'side'}
path_save += f'_{axis_int_to_label[axis_to_save]}'

if filter_preprocess :
    path_save += '_filtered_by_preprocess/'
else :
    path_save += '_all/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Support functions

def filter_files_for_preprocess(file_path_list : list) :
    """
    Inside the ADNI dataset the name of the files contains also the preprocess performend on the data.
    To make sure the data are uniform, I only keep the data that have gone through all 3 preprocessing steps (gradwarp, B1 and N3).
    This function receives a list of file paths and return a list with only the files that have gone through all 3 preprocessing steps.
    """
    
    all_preprocess_list = []
    for file_path in file_path_list :
        tmp_file_path = file_path.lower()

        if 'gradwarp' in tmp_file_path : 
            gradwarp = True
        else : 
            gradwarp = False

        if 'b1' in tmp_file_path : 
            b1 = True
        else : 
            b1 = False
        
        if 'n3' in tmp_file_path : 
            n3 = True
        else : 
            n3 = False

        if gradwarp and b1 and n3 : 
            all_preprocess_list.append(tmp_file_path)
    
    return all_preprocess_list


def get_subj_id_and_scan_date(file_path : str) :
    """
    Given a path to a file, extract the subject ID and the scan date from the file name.
    """

    subj_id = file_path.split('/')[4]
    date = file_path.split('/')[6].split('_')[0]

    return subj_id, date


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Paths for all 3 classes
path_AD = './data/ADNI_1_3YR_3T/AD/'
path_CN = './data/ADNI_1_3YR_3T/CN/'
path_MCI = './data/ADNI_1_3YR_3T/MCI/'

# Get the paths of the files for each class
file_path_list_CN = support_dataset.get_all_files_from_path(path_CN)
file_path_list_AD = support_dataset.get_all_files_from_path(path_AD)
file_path_list_MCI = support_dataset.get_all_files_from_path(path_MCI)

# Filter the files that have gone through all 3 preprocessing steps
if filter_preprocess :
    file_path_list_CN = filter_files_for_preprocess(file_path_list_CN)
    file_path_list_AD = filter_files_for_preprocess(file_path_list_AD)
    file_path_list_MCI = filter_files_for_preprocess(file_path_list_MCI)

# Iterate over class
for file_path_list, class_name in zip([file_path_list_CN, file_path_list_AD, file_path_list_MCI], ['CN', 'AD', 'MCI']) :
    print("Processing class : {}".format(class_name))

    # Iterate over file_path
    for i in range(len(file_path_list)) :
        file_path = file_path_list[i]

        # Get the subject ID and the scan date
        subj_id, date = get_subj_id_and_scan_date(file_path)
        
        # Load the data from the nii file
        data = nib.load(file_path).get_fdata()

        # Create the path to save the images
        path_save_subj = path_save + f'{class_name}/{subj_id}/'
        filename = f'{date}'

        # Convert the 3D matrix to images
        # print(data.shape, file_path.split('/')[5])
        support_dataset.convert_3D_matrix_to_image(data, path_save_subj, filename, axis_to_save, min_idx_to_save, max_idx_to_save)










