"""
Implementation of the nii_to_hdf5 cli function.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

from __future__ import annotations

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

from ..dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def nii_to_hdf5_func(args) -> None:
    """
    Convert all nii files in a folder to a single hdf5 file.
    """
    # ***************************************
    # Check the arguments

    if args.dataset_folder is None:
        raise ValueError("The --dataset_folder argument is required. Please specify the path to the folder with the nii files.")
    elif not os.path.exists(args.dataset_folder):
        raise ValueError(f"The specified dataset folder '{args.dataset_folder}' does not exist. Please specify a valid path to the folder with the nii files.")

    # ***************************************
    # Get the list of nii files to convert
    
    # Get the list of all the nii files in the specified dataset folder (including subfolders)
    nii_files_list = support_dataset.get_all_files_from_path(path_to_explore = args.dataset_folder, filetype_filter = "nii")
    
    # Check that at least one nii file was found in the specified dataset folder
    if len(nii_files_list) == 0:
        raise ValueError(f"No nii files found in the specified dataset folder '{args.dataset_folder}'. Please specify a valid path to the folder with the nii files.")

    # (OPTIONAL) If the --filter argument is specified, keep only the nii files that contain the specified string in their name
    if args.filter is not None:
        nii_files_list = [nii_file for nii_file in nii_files_list if args.filter in os.path.basename(nii_file)]

        # Check that at least one nii file was found in the specified dataset folder that contains the specified string in their name
        if len(nii_files_list) == 0:
            raise ValueError(f"No nii files found in the specified dataset folder '{args.dataset_folder}' that contain the string '{args.filter}' in their name. Please specify a valid path to the folder with the nii files and/or a valid filter string.")

    if args.debug : print(f"Find {len(nii_files_list)} nii files to convert in the specified dataset folder '{args.dataset_folder}'.")

    # ***************************************
    # (OPTIONAL) Load labels

    # Check if the --labels_file argument is specified
    if args.labels_file is not None:
        # Check that the specified labels file exists and is a csv file
        if not os.path.isfile(args.labels_file):
            raise ValueError(f"The specified labels file '{args.labels_file}' does not exist. Please specify a valid path to the labels file.")
        elif not args.labels_file.endswith(".csv"):
            raise ValueError(f"The specified labels file '{args.labels_file}' is not a csv file. Please specify a valid path to the labels file.")
        
        # Load the labels file as a pandas dataframe
        labels_df = pd.read_csv(args.labels_file)

        # Check that the labels file contains the required columns 'file_name' and 'labels_int'
        if "file_name" not in labels_df.columns:
            raise ValueError(f"The specified labels file '{args.labels_file}' does not contain the required column 'file_name'. Please specify a valid path to the labels file. Currently, the columns in the labels file are: {labels_df.columns}.")
        elif "labels_int" not in labels_df.columns:
            raise ValueError(f"The specified labels file '{args.labels_file}' does not contain the required column 'labels_int'. Please specify a valid path to the labels file. Currently, the columns in the labels file are: {labels_df.columns}.")

        # Convert the dataframe to a dictionary with the file names as keys and the labels as values
        if "labels_str" in labels_df.columns:
            labels_dict = {row["file_name"] : {"label_int": row["labels_int"], "label_str": row["labels_str"]} for _, row in labels_df.iterrows()}
        else:
            labels_dict = {row["file_name"] : {"label_int": row["labels_int"]} for _, row in labels_df.iterrows()}

        if args.debug : print(f"Labels file '{args.labels_file}' loaded successfully.")
    else :
        # If the --labels_file argument is not specified, set the labels dictionary to None
        labels_dict = None

        if args.debug : print(f"No labels file specified. No labels will be extracted and the 'dataset_info.csv' file will not be created.")

    # ***************************************
    # Conversion to hdf5

    # Load a single sample to get the shape of the volumes.
    sample_volume_shape = nib.load(nii_files_list[0]).get_fdata().shape

    # Shape is in the formax (X, Y, Z) but we want (1, X, Y, Z) for the hdf5 chunks
    chunk_shape = (1, *sample_volume_shape)
    
    if args.debug : print(f"Sample volume shape : {sample_volume_shape}. hdf5 chunk shape: {chunk_shape}.")

    with h5py.File("dataset.h5", "w") as f:
        # If each sample is shape (D, H, W), e.g. (128, 128, 128)
        ds = f.create_dataset(
            "volumes",
            shape = (len(nii_files_list), *sample_volume_shape),  # (num_samples, X, Y, Z)
            dtype = np.float32,
            chunks = chunk_shape,             # chunk size (1, X, Y, Z)
            compression = "lzf",           # fast, light compression
        )
        for i, path in enumerate(sorted(sample_paths)):
            ds[i] = load_sample(path)    # your existing loader


