"""
Implementation of the cli functions for stats calculation.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from __future__ import annotations

import nibabel as nib
import numpy as np
import os
import torch

from addl.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_mean_and_std(args, file_dataset_extension : str = None) -> None :
    """
    Compute the mean and standard deviation of a dataset stored in a folder.
    To be precise this function will load all the dataset files in the specified folder (and subfolders) with the extension specified in file_dataset_extension, compute the mean and standard deviation of all the data contained in those files, and save them in txt files in the same folder.

    Current supported dataset file extensions are: .npy, .pt, .pth, .nii

    Note that for dataset stored in multiple files (e.g. a dataset stored in multiple .npy files) this function treats each file as a sample and it is up to you to ensure that all the sample files are "correct" (e.g. same shape, same data type, etc.) and that the mean and standard deviation calculated on those files are meaningful for your dataset.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments parsed by the argparse library. The expected arguments are:
        - dataset_folder : str, required, the path to the folder with the dataset files.
        - filter : str, optional, if used keep only the dataset files that contain the specified string in their name. E.g., if --filter is set to 'train', only the dataset files with 'train' in their name will be used for the calculation.
    """

    supported_extensions = [".npy", ".pt", ".pth", ".nii"]

    # ***************************************
    # Check the arguments

    if args.dataset_folder is None:
        raise ValueError("The --dataset_folder argument is required. Please specify the path to the folder with the dataset files.")
    elif not os.path.exists(args.dataset_folder):
        raise ValueError(f"The specified dataset folder '{args.dataset_folder}' does not exist. Please specify a valid path to the folder with the dataset files.")

    if file_dataset_extension is None :
        raise ValueError(f"The file_dataset_extension argument is required. Please specify the extension of the dataset files to use for the calculation. Supported extensions are : {', '.join(supported_extensions)}")
    elif file_dataset_extension not in supported_extensions :
        raise ValueError(f"The specified file_dataset_extension '{file_dataset_extension}' is not supported. Supported extensions are : {', '.join(supported_extensions)}")

    # ***************************************
    # Get the list of dataset files to use for the calculation
    
    # Get the list of dataset files to use for the calculation
    files_list = support_dataset.get_all_files_from_path(path_to_explore = args.files_folder, filetype_filter = file_dataset_extension)

    # (OPTIONAL) Filter the files list using the specified filter string
    if args.filter is not None : files_list = [file for file in files_list if args.filter in file]

    if args.debug :
        print(f"Find {len(files_list)} files in the specified dataset folder '{args.files_folder}'.")
        n_elements_to_print = int(0.05 * len(files_list)) if len(files_list) >= 20 else 2

    # ***************************************
    # Stats calculation

    channel_sum         = 0
    channel_squared_sum = 0
    total_data_point    = 0

    for i in range(len(files_list)) :
        
        # Load the sample file and compute the individual stats for that sample
        channel_sum_sample, channel_squared_sum_sample, total_data_points_sample = __load_sample_file_and_compute_individual_stats(path = files_list[i], extension = file_dataset_extension)

        # Update the global stats with the individual stats of the current sample
        channel_sum         += channel_sum_sample
        channel_squared_sum += channel_squared_sum_sample
        total_data_point    += total_data_points_sample

        # Debug info
        if args.debug and i < n_elements_to_print : print(f"Processing file {i + 1} / {len(files_list)} : '{files_list[i]}'.")
    
    # Compute final mean and std
    final_mean = channel_sum / total_data_point
    final_std = np.sqrt((channel_squared_sum / total_data_point) - (final_mean ** 2))

    # Save the mean
    mean_file_path = os.path.join(args.files_folder, "mean.txt")
    with open(mean_file_path, "w") as mean_file : mean_file.write(str(final_mean))

    # Save the std
    std_file_path = os.path.join(args.files_folder, "std.txt")
    with open(std_file_path, "w") as std_file : std_file.write(str(final_std))


def __load_sample_file_and_compute_individual_stats(path : str, extension : str) :

    # channel_sum += np.sum(sample)
    # channel_squared_sum += np.sum(sample ** 2)
    # total_voxels += sample.shape[0] * sample.shape[1] * sample.shape[2]
    if extension == ".npy" or extension == ".npz" or extension == ".nii" :

        if extension == ".npy" or extension == ".npz" :
            sample = np.load(path)
        elif extension == ".nii" :
            import nibabel as nib
            sample = nib.load(path).get_fdata()

        channel_sum = np.sum(sample)
        channel_squared_sum = np.sum(sample ** 2)
        total_data_point_per_sample = np.prod(sample.shape)

    elif extension in [".pt", ".pth"] :
        import torch
        sample = torch.load(path)

        channel_sum = torch.sum(sample).item()
        channel_squared_sum = torch.sum(sample ** 2).item()
        total_data_point_per_sample = torch.prod(torch.tensor(sample.shape)).item()

    return channel_sum, channel_squared_sum, total_data_point_per_sample

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_hdf5_dataset_length_func(path : str, group_name : str = "data") -> int :
    """
    Get the length of a dataset stored in an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    group_name : str, optional
        The name of the group in the hdf5 file where the dataset is stored. Default is "data". Note that the dataset must be stored in a group with the specified name.

    Returns
    -------
    int
        The length of the dataset.
    """

    # TODO : Consider if move it inside the dataset module

    import h5py

    with h5py.File(path, "r") as f : dataset_length = f["data"].shape[0]

    return dataset_length





