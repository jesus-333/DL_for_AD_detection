"""
Implementation of the nii_to_hdf5 cli function.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import numpy as np
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
        labels_dict = dict()
        for _, row in labels_df.iterrows() :
            # Create a temporary dictionary to store the labels for the current file
            tmp_entry = dict()
            
            # Add the required labels to the temporary dictionary
            tmp_entry["label_int"] = row["labels_int"]
            
            # (OPTIONAL) Add the labels in string format and the subject id to the temporary dictionary if they are present in the labels file
            if "labels_str" in labels_df.columns : tmp_entry["label_str"] = row["labels_str"]
            if "subj_id" in labels_df.columns    : tmp_entry["subj_id"]   = row["subj_id"]
            
            # Add the temporary dictionary to the labels dictionary with the file name as key
            labels_dict[os.path.basename(row["file_name"])] = tmp_entry

        if args.debug : print(f"Labels file '{args.labels_file}' loaded successfully.")
    else :
        # If the --labels_file argument is not specified, set the labels dictionary to None
        labels_dict = None
        if args.debug : print("No labels file specified. No labels will be extracted and the 'dataset_info.csv' file will not be created.")

    # ***************************************
    # Conversion to hdf5

    # Load a single sample to get the shape of the volumes.
    sample_volume_shape = nib.load(nii_files_list[0]).get_fdata().shape

    # Shape is in the formax (X, Y, Z) but we want (1, X, Y, Z) for the hdf5 chunks
    chunk_shape = (1, *sample_volume_shape)
    if args.debug : print(f"Sample volume shape : {sample_volume_shape}. hdf5 chunk shape : {chunk_shape}.")

    columns = ['subj_id', 'labels_int', 'labels_str', 'file_name']
    dataset_info_csv  = pd.DataFrame(columns = columns)

    # Check if the output directory flag was specified, if not use the dataset folder as output directory
    if args.output_folder is not None :
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok = True)
    else :
        output_folder = args.dataset_folder
    if args.debug : print(f"Output directory for the hdf5 file and the dataset_info.csv file: '{output_folder}'.")
    
    # Crate path(s) for output file (and dataset info csv if labels are provided)
    path_dataset_file = os.path.join(output_folder, "dataset.h5")
    if labels_dict is not None : path_dataset_info_csv = os.path.join(output_folder, "dataset_info.csv")

    with h5py.File(path_dataset_file, "w") as f :
        # Create the hdf5 dataset
        ds = f.create_dataset(
            "data",
            shape       = (len(nii_files_list), *sample_volume_shape),  # (num_samples, X, Y, Z)
            dtype       = np.float32,  # Convert to float32 to save space
            chunks      = chunk_shape, # chunk size (1, X, Y, Z)
            compression = "lzf",       # fast, light compression
        )

        # Iterate over the nii files and save them in the hdf5 file
        for i in range(len(nii_files_list)) :
            path_sample = nii_files_list[i]

            # Get sample in float32
            sample = nib.load(path_sample).get_fdata().astype(np.float32)
            
            # (OPTIONAL) Apply min-max normalization to the sample
            if args.minmax : sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))

            # (OPTIONAL) Apply z-score normalization to the sample
            if args.z_score : sample = (sample - sample.mean()) / sample.std()
            
            # Saved the sample in the hdf5 file
            ds[i] = sample

            # (OPTIONAL) If the labels dictionary is not None, extract the label for the current sample and save it in the dataset_info_csv dataframe
            if labels_dict is not None :
                file_name = os.path.basename(path_sample)
                # Row for the current sample
                tmp_info_csv = []

                # Add information for the current sample
                tmp_info_csv.append(labels_dict[file_name]["subj_id"] if "subj_id" in labels_dict[file_name] else "-")
                tmp_info_csv.append(labels_dict[file_name]["label_int"])
                tmp_info_csv.append(labels_dict[file_name]["label_str"] if "label_str" in labels_dict[file_name] else "-")
                tmp_info_csv.append(file_name)

                # Add the row for the current sample to the dataset_info_csv dataframe
                tmp_info_csv = np.asarray(tmp_info_csv)
                dataset_info_csv = pd.concat([dataset_info_csv, pd.DataFrame(tmp_info_csv.reshape(1, -1), columns = columns)], ignore_index = True)

        if args.debug : print(f"Conversion completed successfully. hdf5 file saved at '{path_dataset_file}'.")

        # Save the dataset_info_csv dataframe as a csv file in the same folder as the hdf5 file
        if labels_dict is not None : 
            dataset_info_csv.to_csv(path_dataset_info_csv, index = False)
            if args.debug : print(f"Dataset info csv file saved at '{path_dataset_info_csv}'.")







