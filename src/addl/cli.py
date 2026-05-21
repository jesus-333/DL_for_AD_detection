"""
Implementation the command-line interface (cli)

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from __future__ import annotations

import argparse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def nii_to_hdf5() -> None:
    """
    Convert all nii files in a folder to a single hdf5 file.
    """
    
    from .cli_script import nii_to_hdf5

    # ***************************************
    # Command-line arguments parsing
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(prog = "nii-to-hdf5", description = "Convert all nii files in a folder to a single hdf5 files.")
    parser.add_argument("--dataset_folder", default = None, help = "Path to the folder with the nii files. Note that all the nii files inside this path will be converted to hdf5 files. This means both in the root folder and in all the subfolders.")
    parser.add_argument("--filter"        , default = None, help = "If used keep only the nii files that contain the specified string in their name. E.g., if --filter is set to 'train', only the nii files with 'train' in their name will be converted to hdf5 files.")
    parser.add_argument("--output_folder" , default = None, help = "Path to the folder where the hdf5 files will be saved. If not specified, the hdf5 files will be saved in the same folder as the nii files.")
    parser.add_argument("--labels_file"   , default = None, help = "Path to the file with the labels. This file must be a csv file with at least two columns 'file_name' (i.e. the name of the file with extension) and 'labels_int' (the label of the sample contained in the file, stored as an int). An optional two columns, 'labels_str' (the label of the sample contained in the file, stored as an str) and 'subj_id' (the id of patient of the sample) coulb be present . During the conversion, the labels will be extracted from this file by matching the 'file_name' column with the name of the nii files. If passed together with hd5f file,  it will created also a 'dataset_info.csv' file in the output folder, which will contain for each sample in the hdf5 file the corresponding label, extracted from the labels file. If not specified, no labels will be extracted and the 'dataset_info.csv' file will not be created.")
    parser.add_argument("--minmax"        , default = False, action = "store_true", help = "If used, the function will apply a min-max normalization to the data before saving it in the hdf5 file. Note that the normalization will be applied sample-wise")
    parser.add_argument("--z_score"       , default = False, action = "store_true", help = "If used, the function will apply a z-score normalization to the data before saving it in the hdf5 file. Note that the normalization will be applied sample-wise")
    parser.add_argument("--compute_stats" , default = False, action = "store_true", help = "If used, mean and std of the dataset will be computed during the conversion and saved in npy files. Note that if a normalization is applied to the data, the computed mean and std will be the ones of the normalized data.")
    parser.add_argument("--debug"         , default = False, action = "store_true", help = "If used, the function will print some debug information during the conversion process. This can be useful to understand what is happening during the conversion and to identify potential issues.")
    args = parser.parse_args()

    # ***************************************

    # Call the function that implements the cli command
    nii_to_hdf5.nii_to_hdf5_func(args)


def pt_to_npy() -> None:
    """
    Convert all pt (or pth) files in a folder to a single npy file.
    """
    
    from .cli_script import pt_to_npy

    # ***************************************
    # Command-line arguments parsing
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(prog = "pt-to-npy", description = "Convert all pt files in a folder to a single npy file.")
    parser.add_argument("--files_folder", default = None, help = "Path to the folder with the pt files. Note that all the pt files inside this path will be converted to npy files. This means both in the root folder and in all the subfolders.")
    parser.add_argument("--filter"        , default = None, help = "If used keep only the pt files that contain the specified string in their name. E.g., if --filter is set to 'train', only the pt files with 'train' in their name will be converted to npy files.")
    parser.add_argument("--output_folder" , default = None, help = "Path to the folder where the npy files will be saved. If not specified, the npy files will be saved in the same folder as the pt files.")
    parser.add_argument("--debug"         , default = False, action = "store_true", help = "If used, the function will print some debug information during the conversion process. This can be useful to understand what is happening during the conversion and to identify potential issues.")
    args = parser.parse_args()

    # ***************************************

    # Call the function that implements the cli command
    pt_to_npy.pt_to_npy_func(args)

def npy_to_pt() -> None:
    """
    Convert all npy files in a folder to a single pt file.
    """
    
    from .cli_script import pt_to_npy

    # ***************************************
    # Command-line arguments parsing
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(prog = "npy-to-pt", description = "Convert all npy files in a folder to a single pt file.")
    parser.add_argument("--files_folder", default = None, help = "Path to the folder with the npy files. Note that all the npy files inside this path will be converted to pt files. This means both in the root folder and in all the subfolders.")
    parser.add_argument("--filter"        , default = None, help = "If used keep only the npy files that contain the specified string in their name. E.g., if --filter is set to 'train', only the npy files with 'train' in their name will be converted to pt files.")
    parser.add_argument("--output_folder" , default = None, help = "Path to the folder where the pt files will be saved. If not specified, the pt files will be saved in the same folder as the npy files.")
    parser.add_argument("--debug"         , default = False, action = "store_true", help = "If used, the function will print some debug information during the conversion process. This can be useful to understand what is happening during the conversion and to identify potential issues.")
    args = parser.parse_args()

    # ***************************************

    # Call the function that implements the cli command
    pt_to_npy.pt_to_npy(args, reverse = True)
