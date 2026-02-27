"""
Conceptually it works similarly to create_idx_files_for_federated_simulations.py but instead of dividing the data uniformly between the clients, it divides the patient uniformly between the clients.
The output of this script will be indices files, saved in the same way as in create_idx_files_for_federated_simulations.py. So, for more details on the output of the script, please refer to the header of that script.

The parameters of the script are almost identical to the ones of create_idx_files_for_federated_simulations.py, the only difference are :
- --percentage_data_used_for_training works slightly differently than in create_idx_files_for_federated_simulations.py. Here we count the number of samples per subject, we compute how many samples we need for training and then we try to have roughly that number of samples in the training set (by putting enough patients in the training set in order to reach at least the required amount)
- --keep_labels_proportion was removed
- --keep_samples_proportion was added to allow the user to specify if the split of the data between clients should be done while trying to keep the same proportion of samples for clients (i.e. all client have roughly the same number of samples)

Note that this script can create clients with a very different number of samples if keep_samples_proportion is set to False.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (OPTIONAL) CLI Arguments and check

import argparse

parser = argparse.ArgumentParser(description = 'Compute indices files for federated simulations from a dataset tensor file.')
parser.add_argument('--path_data'                        , type = str  , default = None, help = 'Path to the folder with the data. If not provided, it will use the value defined in this script.')
parser.add_argument('--name_tensor_file'                 , type = str  , default = None, help = 'Name of the tensor file with the dataset. If not provided, it will use the value defined in this script.')
parser.add_argument('--path_to_save'                     , type = str  , default = None, help = 'Path to the folder where the dataset will be saved. If not specified it will use the value defined in this script. If the folder does not exist it will be created.')
parser.add_argument('--percentage_data_used_for_training', type = float, default = None, help = 'Percentage of subjects to use for training (the rest will be used for validation). If not specified it will use the value defined in this script.')
parser.add_argument('--num_clients'                          , type = int  , default = None, help = 'Number of clients to simulate. If not specified it will use the value defined in this script.')
parser.add_argument('--seed'                                 , type = int  , default = None, help = 'Random seed for reproducibility. It must be a positive integer. If not specified it will use the value defined in this script.')
parser.add_argument('--n_folds'                              , type = int  , default = None, help = 'Number of folds for cross fold validation. If not specified it will use the value defined in this script.')

# Boolean arguments
parser.add_argument('--use_cross_fold_validation'            , default = None, action = 'store_true', help = 'If passed the script will use cross fold validation. See the script header for more details.')
parser.add_argument('--keep_samples_proportion'              , default = None, action = 'store_true', help = 'If passed as an argument, the split of the data between clients will be done while trying to keep the same proportion of samples for each class in each client as in the original dataset. If not passed as an argument, the split will be done only dividing the patients uniformly between clients')

# Negate boolean arguments
parser.add_argument('--no-use_cross_fold_validation'     , action = 'store_false', dest = 'use_cross_fold_validation', help = 'If passed as an argument, no cross-fold validation will be applied')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pandas as pd
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Arguments

path_data = "data/ADNI_axial_middle_slice/"
name_tensor_file = "dataset_tensor___176_resize.pt"
dataset_info_file = "dataset_info.csv"
path_to_save = "data/ADNI_axial_middle_slice/FL_indices/"

percentage_data_used_for_training = 0.8
num_clients = 15
seed = -1
n_folds = None # If None, it will be computed from percentage_data_used_for_training

use_cross_fold_validation = False
keep_samples_proportion = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check arguments

# Check if arguments are passed through the command line
# If the arguments are passed through the command line, they will overwrite the ones set inside the script
if args.path_data is not None                         : path_data = args.path_data
if args.name_tensor_file is not None                  : name_tensor_file = args.name_tensor_file
if args.path_to_save is not None                      : path_to_save = args.path_to_save
if args.percentage_data_used_for_training is not None : percentage_data_used_for_training = args.percentage_data_used_for_training
if args.num_clients is not None                           : num_clients = args.num_clients
if args.seed is not None                              : seed = args.seed
if args.n_folds is not None                           : n_folds = args.n_folds
if args.use_cross_fold_validation is not None         : use_cross_fold_validation = args.use_cross_fold_validation
if args.keep_samples_proportion is not None         : keep_samples_proportion = args.keep_samples_proportion

# Check arguments validity
if seed <= 0 :
    seed = np.random.randint(0, 2**32 - 1)
    print(f"Invalid seed value. Sample new random seed {seed}")

if n_folds is None :
    if not (0.0 < percentage_data_used_for_training < 1.0) :
        if use_cross_fold_validation : print("If use_cross_fold_validation is True and n_folds is not specified, the number of folds will be computed from percentage_data_used_for_training.")
        raise ValueError(f"percentage_data_used_for_training must be between 0 and 1. Current value: {percentage_data_used_for_training}")

possible_values_percentage_data_used_for_training = [0.5, 0.75, 0.8, 0.9, 0.95, 0.98]
if percentage_data_used_for_training not in possible_values_percentage_data_used_for_training and use_cross_fold_validation and n_folds is None :
    print(f"Warning: percentage_data_used_for_training is set to {percentage_data_used_for_training} while use_cross_fold_validation is True and n_folds is not specified")
    print(f"In this case, it is recommended to use one of the standard values: {possible_values_percentage_data_used_for_training}")
    print("percentage_data_used_for_training will be rounded to the closest standard value.")
    percentage_data_used_for_training = min(possible_values_percentage_data_used_for_training, key = lambda x : abs(x - percentage_data_used_for_training))
    print(f"New value for percentage_data_used_for_training: {percentage_data_used_for_training}")

# Create the folder to save the indices files
os.makedirs(path_to_save, exist_ok = True)

np.random.seed(seed)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_train_val_split(subj_list : list, sample_per_subj : dict, min_sample_training : int) -> tuple :
    """
    Divide the subjects between the training and validation sets. The training set will contain at least min_sample_training samples, and the remaining subjects will be put in the validation set.
    Note that this function does not guarantee, per se, that the training set will contain exactly min_sample_training samples, but only that it will contain at least that number of samples.
    
    Note that, if you want to use this function to put at least 80% of the data in the training set, you can set min_sample_training to 0.8 * n_samples, where n_samples is the total number of samples in the dataset. 
    In this way, the function will put subjects in the training set until it reaches at least 80% of the samples, and then it will put the remaining subjects in the validation set.
    This method is valid for any percentage you want.

    Parameters
    ----------
    subj_list : list
        List of unique subjects in the dataset.
    sample_per_subj : dict
        Dictionary with the number of samples for each subject. The keys are the subject IDs and the values are the number of samples for each subject.
        Note that the subject IDs in sample_per_subj must be the same as the ones in subj_list.
    min_sample_training : int
        Minimum number of samples to put in the training set.

    Returns
    -------
    list_subj_train : list
        List of subjects in the training set.
    list_subj_val : list
        List of subjects in the validation set.
    """
    
    # Shuffle the subjects list
    np.random.shuffle(subj_list)
    
    # Create the training and validation lists of subjects
    list_subj_train = []
    list_subj_val = []
    samples_in_training_data = 0
    
    # Loop over the subjects and put them in the training set until I have enough samples in the training data, then put the remaining subjects in the validation set
    for i in range(len(subj_list)) :
        current_subj = subj_list[i]
        if samples_in_training_data >= min_sample_training :
            # If I have already enough samples in the training data, I put all the remaining subjects in the validation set
            list_subj_val.append(current_subj)
        else :
            # If I don't have enough samples in the training data, I put the current subject in the training set
            list_subj_train.append(current_subj)
            samples_in_training_data += sample_per_subj[subj]

    return list_subj_train, list_subj_val

def divide_subj_per_client_uniformly(subj_list : list, num_clients : int) -> tuple :
    """
    Divide the subjects uniformly between the clients. 
    Each client will have roughly the same number of subjects, but not necessarily the same number of samples (since different subjects can have a different number of samples).
    """
    
    # Divide the subjects uniformly between the clients
    np.random.shuffle(list_subj_train)
    subj_per_client = np.array_split(list_subj_train, num_clients)

    # Count the number of samples for each client
    samples_per_client = np.zeros(num_clients, dtype = int)
    for i in range(num_clients) :
        for subj in subj_per_client[i] :
            samples_per_client[i] += sample_per_subj[subj]

    return subj_per_client, samples_per_client

def divide_subj_per_client_keeping_sample_proportion(subj_list : list, sample_per_subj : dict, num_clients : int) -> tuple :
    """
    Divide the subjects between the clients while trying to keep the same number of samples for each client.
    Note that the function does not guarantee that the number of samples for each client will be exactly the same, but it tries to keep it as balanced as possible.
    """
    
    # Each element of the list will contain the list of subjects for the corresponding client
    subj_per_client = [[] for _ in range(num_clients)]
    
    # Contains the number of samples assigned to each client
    samples_per_client = np.zeros(num_clients, dtype = int)

    for i in range(len(subj_list)) :
        client_with_less_samples = np.argmin(samples_per_client)

        current_subj = subj_list[i]
        subj_per_client[client_with_less_samples].append(current_subj)
        samples_per_client[client_with_less_samples] += sample_per_subj[current_subj]

    return subj_per_client, samples_per_client

def create_idx_array_for_subj(subj, subj_list_per_sample : list) :
    """
    Create an array with all the indices of the samples of a specific subject

    (Yeah, I know, this function is very simple, but it help me to keep the code more organized)

    Parameters
    ----------
    subj : str 
        Subject ID for which to create the indices array.
        Note that I specified the type as str because I save the subject IDs as strings in the dataset_info.csv file, but it can be of any type as long as it is consistent with the type of the subject IDs in the dataset_info.csv file.
    subj_list_per_sample : list
        List with the subject ID for each sample in the dataset. The length of this list is equal to the total number of samples in the dataset, and the i-th element of the list is the subject ID of the i-th sample in the dataset.

    Returns
    -------
    idx_array : np.array
        Array with all the indices of the samples of the specified subject.
    """

    idx_array = np.where(subj_list_per_sample == subj)[0]
    return idx_array

def create_idx_array_for_client(subj_list_client : list, subj_list_per_sample : list) :
    """
    Create an array with all the indices of the samples of a specific client, given the list of subjects assigned to that client.

    Parameters
    ----------
    subj_list_client : list
        List of subject IDs assigned to the client.
    subj_list_per_sample : list
        List with the subject ID for each sample in the dataset. The length of this list is equal to the total number of samples in the dataset, and the i-th element of the list is the subject ID of the i-th sample in the dataset.

    Returns
    -------
    idx_array : np.array
        Array with all the indices of the samples of the specified client.
    """

    idx_array = np.array([], dtype = int)
    for subj in subj_list_client :
        idx_subj = create_idx_array_for_subj(subj, subj_list_per_sample)
        idx_array = np.concatenate((idx_array, idx_subj))

    return idx_array

def check_idx_array(idx_array_list, n_samples : int) :
    """
    Check that the indices arrays in idx_array_list are valid, i.e. that they do not contain duplicates and that they do not contain indices out of range.

    Parameters
    ----------
    idx_array_list : list
        List of arrays with the indices of the samples for each client and for the validation set.
    n_samples : int
        Total number of samples in the dataset.

    Raises
    ------
    ValueError
        If there are duplicate indices in the arrays or if there are indices out of range.
    """

    # Check that there are no duplicate indices in the arrays
    all_indices = np.concatenate(idx_array_list)
    unique_indices = np.unique(all_indices)
    if len(all_indices) != len(unique_indices) :
        raise ValueError("There are duplicate indices in the arrays.")

    # Check that there are no indices out of range
    if np.any(all_indices < 0) or np.any(all_indices >= n_samples) :
        raise ValueError("There are indices out of range in the arrays.")

    print("The indices arrays are valid. No duplicates and no indices out of range.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

data = torch.load(f'{path_data}{name_tensor_file}', mmap = True)
n_samples = len(data)
print(f"Total number of samples in the dataset : {len(data)}")
del data

dataset_info = pd.read_csv(f'{path_data}{dataset_info_file}')
subj_list_per_sample = dataset_info['subj_id'].to_numpy()
all_idx = np.arange(len(subj_list_per_sample))
print(f"Total number of unique subjects in the dataset : {len(np.unique(subj_list_per_sample))}\n")

# Count samples per subject
sample_per_subj = {}
subj_list, counts_per_subj = np.unique(subj_list_per_sample, return_counts = True)
for subj, count in zip(subj_list, counts_per_subj) : sample_per_subj[subj] = count

# Compute the minimum number of samples for training
min_sample_training = int(n_samples * percentage_data_used_for_training)

if use_cross_fold_validation :
    pass
else :
    # Divide the subjects between the training and validation sets
    list_subj_train, list_subj_val = create_train_val_split(subj_list, sample_per_subj, min_sample_training)
    
    # Divide the training subjects between the clients
    if keep_samples_proportion :
        subj_per_client_train, samples_per_client_train = divide_subj_per_client_keeping_sample_proportion(list_subj_train, sample_per_subj, num_clients)
    else :
        subj_per_client_train, samples_per_client_train = divide_subj_per_client_uniformly(list_subj_train, num_clients)
    
    # Print the number of subjects and samples for each client and for the validation set
    print("Training subjects per client:")
    for i in range(num_clients) : print(f"\tClient {i} : {len(subj_per_client_train[i])} subjects, {samples_per_client_train[i]} samples")
    print(f"Validation subjects {len(list_subj_val)} subjects, {n_samples - sum(samples_per_client_train)} samples")

    # Used to check that the indices arrays do not contain duplicates and that they do not contain indices out of range
    all_idx_array = []

    # Create the indices files for the training
    for i in range(num_clients) :
        idx_array_client = create_idx_array_for_client(subj_per_client_train[i], subj_list_per_sample)
        np.save(f'{path_to_save}train_idx_client_{i}.npy', idx_array_client)
        all_idx_array.append(idx_array_client)

    # Create the indices file for the validation set
    idx_array_val = create_idx_array_for_client(list_subj_val, subj_list_per_sample)
    np.save(f'{path_to_save}val_idx.npy', idx_array_val)
    all_idx_array.append(idx_array_val)

    # Check that the indices arrays are valid
    check_idx_array(all_idx_array, n_samples)

print("Data splitting and saving of indices files completed.")
