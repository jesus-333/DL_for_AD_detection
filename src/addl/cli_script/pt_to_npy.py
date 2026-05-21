"""
Implementation of the pt to npy function and reverse

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import numpy as np
import torch
import os
import pandas as pd

try :
    import h5py
except ImportError :
    raise ImportError("h5py library is not installed. Please install it to use the dataset_HDF5 class.")

try :
    import nibabel as nib
except ImportError :
    raise ImportError("nibabel library is not installed. Please install it to use the nii_to_hdf5 function.")

from addl.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def pt_to_npy(args, reverse : bool = False) -> None:
    """
    Convert all pt (or pth) file in a folder to a single npy file. If reverse is True, convert all npy files in a folder to pt files.

    Note that all the pt (or npy) files inside the specified dataset folder will be converted to npy (or pt) files. This means both in the root folder and in all the subfolders.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed by the argparse library. The expected arguments are:
        - files_folder : str, required, the path to the folder with the pt files.
        - filter : str, optional, if used keep only the pt files that contain the specified string in their name. E.g., if --filter is set to 'train', only the pt files with 'train' in their name will be converted to npy files.
        - output_folder : str, optional, the path to the folder where the npy files will be saved. If not specified, the npy files will be saved in the same folder as the pt files.
    """
    # ***************************************
    # Check the arguments

    if args.files_folder is None:
        raise ValueError("The --files_folder argument is required. Please specify the path to the folder with the nii files.")
    elif not os.path.exists(args.files_folder):
        raise ValueError(f"The specified files folder '{args.files_folder}' does not exist. Please specify a valid path to the folder with the nii files.")

    # ***************************************
    # Get the list of pt/pth (or npy) files to convert
    
    # Get the list of all the files in the specified folder (including subfolders)
    if reverse :
        files_list = support_dataset.get_all_files_from_path(path_to_explore = args.files_folder, filetype_filter = "npy")
    else :
        files_list_1 = support_dataset.get_all_files_from_path(path_to_explore = args.files_folder, filetype_filter = "pt")
        files_list_2 = support_dataset.get_all_files_from_path(path_to_explore = args.files_folder, filetype_filter = "pth")
        files_list = files_list_1 + files_list_2
    
    # Check that at least one nii file was found in the specified dataset folder
    if len(files_list) == 0 :
        raise ValueError(f"No files found in the specified dataset folder '{args.files_folder}'. Please specify a valid path to the folder with the files.")

    # (OPTIONAL) If the --filter argument is specified, keep only the nii files that contain the specified string in their name
    if args.filter is not None:
        files_list = [tmp_file for tmp_file in files_list if args.filter in os.path.basename(tmp_file)]

        # Check that at least one nii file was found in the specified dataset folder that contains the specified string in their name
        if len(files_list) == 0 :
            raise ValueError(f"No nii files found in the specified dataset folder '{args.files_folder}' that contain the string '{args.filter}' in their name. Please specify a valid path to the folder with the files and/or a valid filter string.")

    if args.debug :
        print(f"Find {len(files_list)} nii files to convert in the specified dataset folder '{args.files_folder}'.")
        n_elements_to_print = int(0.05 * len(files_list)) if len(files_list) >= 20 else 2

    # ***************************************
    # Conversion

    # Check if the output directory flag was specified, if not use the dataset folder as output directory
    if args.output_folder is not None :
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok = True)
    else :
        output_folder = args.files_folder
    if args.debug : print(f"Output directory for the converted files : '{output_folder}'.")

    for i in range(len(files_list)) :
        path_current_file = files_list[i]
        if reverse : # If reverse is True, convert npy to pt
            # Read the npy file
            tmp_data = np.load(path_current_file)
            
            # Convert the data to a torch tensor
            tmp_data = torch.from_numpy(tmp_data)

            # Save the data in a pt files
            output_file_name = os.path.basename(path_current_file).replace('.npy', '.pt')
            otput_file_path = os.path.join(output_folder, output_file_name)
            torch.save(tmp_data, otput_file_path)

        else : # If reverse is False, convert pt(h) to npy
            # Load the data and convert it to a numpy array
            tmp_data = torch.load(files_list[i], map_location = 'cpu').numpy()

            # Save the data in a npy file
            if 'pth' in path_current_file :
                output_file_name = os.path.basename(path_current_file).replace('.pth', '.npy')
            else :
                output_file_name = os.path.basename(path_current_file).replace('.pt', '.npy')
            otput_file_path = os.path.join(output_folder, output_file_name)
            np.save(otput_file_path, tmp_data)

        # (OPTIONAL) Print some debug information about the conversion process
        if args.debug and (i % n_elements_to_print == 0 or i == len(files_list) - 1) :
            print(f"Converted {i + 1}/{len(files_list)} files. Current file : '{path_current_file}'.")










