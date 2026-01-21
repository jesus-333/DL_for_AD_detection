"""
To work this script requires the pt files created with the create_dataset_pt_files.py script

The data will be firstly split in two parts :
- training data (used for federated learning simulation or centralized training)
- validation data (used for evaluation on both federated learning simulation and centralized training)
The split will be done according to the percentage_data_used_for_training parameter in the arguments.

The training data will be then split uniformly between n clients.

At the end of the script, the files will be saved in the folder defined by the path_to_save argument. If the folder does not exist it will be created.
Since the data will be used for a simulation I decided to save only the indices (respect the original pt file) to reduce the space used on disk.
The folder will contain the following files:
- train_idx_client_0.npy : indices of the training data for client 0
- train_idx_client_1.npy : indices of the training data for client 1
- ...
- train_idx_client_{num_clients - 1}.npy : indices of the training data for last client
- val_idx.npy : indices of the validation data
- train_idx_all.npy : indices of the training data (all clients) (basically the concatenation of all train_idx_client_{i}.npy files)

E.g. if num_clients is 3 and percentage_data_used_for_training is 0.8, the folder will contain the following files:
- path_to_save/
    - train_idx_client_0.npy    (data for client 0, roughly 26.67% of the data)
    - train_idx_client_1.npy    (data for client 1, roughly 26.67% of the data)
    - train_idx_client_2.npy    (data for client 2, roughly 26.67% of the data)
    - val_idx.npy               (20% of the data)
    - train_idx_all.npy         (80% of the data)

If use_cross_fold_validation is set to True, the training data will be split in n_folds folds for cross fold validation. In this case, there are two possibilities :
1) You provide the n_folds argument directly.
2) You do not provide the n_folds argument. In this case, the n_folds will be computed from percentage_data_used_for_training. The number of folds (n_folds) will be computed as n_folds = 1 / (1 - percentage_data_used_for_training).
Note that for option 1, the provided percentage_data_used_for_training will be ignored and the actual percentage of data used for training will be computed as percentage_data_used_for_training = (n_folds - 1) / n_folds.
Note that for option 2 instead, the percentage_data_used_for_training will be rounded to the closest standard value (0.5, 0.75, 0.8, 0.9, 0.95, 0.98) to ensure that n_folds is an integer.
E.g. if percentage_data_used_for_training is 0.8, the number of folds will be 5 (1 / (1 - 0.8) = 1 / 0.2 = 5).
In case of cross fold validation, the data will be saved in subfolders named fold_0, fold_1, ..., fold_{n_folds - 1}.
Inside each folder, the files will be named as described before.

E.g. if num_clients is 3, percentage_data_used_for_training is 0.75 and use_cross_fold_validation is True and you have not provide n_folds as argument, the final output of this script will be the following structure:
- path_to_save/
    - fold_0/
        - train_idx_client_0.npy    (data for client 0, roughly 26.67% of the data)
        - train_idx_client_1.npy    (data for client 1, roughly 26.67% of the data)
        - train_idx_client_2.npy    (data for client 2, roughly 26.67% of the data)
        - val_idx.npy               (20% of the data)
        - train_idx_all.npy         (80% of the data)
    - fold_1/
        - train_idx_client_0.npy    (data for client 0, roughly 26.67% of the data)
        - train_idx_client_1.npy    (data for client 1, roughly 26.67% of the data)
        - train_idx_client_2.npy    (data for client 2, roughly 26.67% of the data)
        - val_idx.npy               (20% of the data)
        - train_idx_all.npy         (80% of the data)
    - fold_2/
        - train_idx_client_0.npy    (data for client 0, roughly 26.67% of the data)
        - train_idx_client_1.npy    (data for client 1, roughly 26.67% of the data)
        - train_idx_client_2.npy    (data for client 2, roughly 26.67% of the data)
        - val_idx.npy               (20% of the data)
        - train_idx_all.npy         (80% of the data)
    - fold_3/
        - train_idx_client_0.npy    (data for client 0, roughly 26.67% of the data)
        - train_idx_client_1.npy    (data for client 1, roughly 26.67% of the data)
        - train_idx_client_2.npy    (data for client 2, roughly 26.67% of the data)
        - val_idx.npy               (20% of the data)
        - train_idx_all.npy         (80% of the data)

The arguments can be passed through the command line or set inside the script. If used, the argument from the command line will overwrite the ones set inside the script.
Note that addl is the name of the package built through hatch from the file in the src folder.

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
parser.add_argument('--percentage_data_used_for_training', type = float, default = None, help = 'Percentage of data to use for training (the rest will be used for validation). If not specified it will use the value defined in this script.')
parser.add_argument('--num_clients'                         , type = int  , default = None, help = 'Number of clients to simulate. If not specified it will use the value defined in this script.')
parser.add_argument('--seed'                             , type = int  , default = None, help = 'Random seed for reproducibility. It must be a positive integer. If not specified it will use the value defined in this script.')
parser.add_argument('--n_folds'                          , type = int  , default = None, help = 'Number of folds for cross fold validation. If not specified it will use the value defined in this script.')

# Boolean arguments
parser.add_argument('--use_cross_fold_validation'        , default = None, action = 'store_true', help = 'If passed the script will use cross fold validation. See the script header for more details.')
parser.add_argument('--keep_labels_proportion'           , default = None, action = 'store_true', help = 'If passed, the data will be split between clients trying to keep the same label proportion in each client as in the original dataset.')

# Negate boolean arguments
parser.add_argument('--no-use_cross_fold_validation'     , action = 'store_false', dest = 'use_cross_fold_validation', help = 'If passed as an argument, no cross-fold validation will be applied')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pandas as pd
import torch

from addl.federated import support_federated_generic

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Arguments

path_data = "data/ADNI_axial_middle_slice/"
name_tensor_file = "dataset_tensor___176_resize.pt"
path_to_save = "data/ADNI_axial_middle_slice/FL_idx/"

percentage_data_used_for_training = 0.8
num_clients = 5
seed = -1
n_folds = None # If None, it will be computed from percentage_data_used_for_training

use_cross_fold_validation = True
keep_labels_proportion = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check arguments

# Check if arguments are passed through the command line
# If the arguments are passed through the command line, they will overwrite the ones set inside the script
if args.path_data is not None                         : path_data = args.path_data
if args.name_tensor_file is not None                  : name_tensor_file = args.name_tensor_file
if args.path_to_save is not None                      : path_to_save = args.path_to_save
if args.percentage_data_used_for_training is not None : percentage_data_used_for_training = args.percentage_data_used_for_training
if args.num_clients is not None                          : num_clients = args.num_clients
if args.seed is not None                              : seed = args.seed
if args.n_folds is not None                           : n_folds = args.n_folds
if args.use_cross_fold_validation is not None         : use_cross_fold_validation = args.use_cross_fold_validation
if args.keep_labels_proportion is not None            : keep_labels_proportion = args.keep_labels_proportion

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def split_and_save_indices(idx_train, idx_val, labels_train, path_to_save : str, print_info : bool = True) :
    """
    Split the training data between clients and save the indices files (both training and validation).
    The function was created to avoid code duplication when using cross fold validation.
    """

    # Get labels for training data
    labels_train = all_labels_int[idx_train]

    # Print info (check if the total number of samples is correct)
    if print_info :
        print("Split training / validation:")
        print(f"Number of training samples  : {len(idx_train)}")
        print(f"Number of validation samples: {len(idx_val)}")
        print(f"Total number of samples     : {len(idx_train) + len(idx_val)}\n")

    idx_clients_train, _ = support_federated_generic.split_data_for_clients_uniformly(idx_train, num_clients = num_clients,
                                                                                      seed = seed,
                                                                                      labels = labels_train, keep_labels_proportion = keep_labels_proportion
                                                                                      )

    # Print info (check if the total number of samples is correct)
    if print_info :
        print(f"Split training data between {num_clients} clients:")
        for i in range(num_clients) : print(f"\tNumber of training samples for client {i}: {len(set(idx_clients_train[i]))}")
        print(f"Total number of training samples: {sum([len(idx_clients_train[i]) for i in range(num_clients)])}")

    # Save the indices files
    np.save(f'{path_to_save}val_idx.npy', idx_val)
    np.save(f'{path_to_save}train_idx_all.npy', idx_train)
    for i in range(num_clients) : np.save(f'{path_to_save}train_idx_client_{i}.npy', idx_clients_train[i])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

data = torch.load(f'{path_data}{name_tensor_file}', mmap = True)
all_idx = np.arange(len(data))
print(f"Total number of samples in the dataset : {len(all_idx)}\n")

dataset_info = pd.read_csv(f'{path_data}dataset_info.csv')
all_labels_int = dataset_info['labels_int'].to_numpy()

if use_cross_fold_validation :
    # (OPTIONAL) Compute the number of folds.
    # The int function is used to avoid problems with floating point precision (e.g. 1 / (1 - 0.98) = 49.99999999999996 on my machine)
    if n_folds is None : n_folds = int(1 / (1 - percentage_data_used_for_training)) if n_folds is None else n_folds

    # Split the data uniformly between folds
    # Note that this function was originally designed to split data (uniformly) between clients, but here we can exploit it to split data uniformly between folds
    idx_per_fold, _ = support_federated_generic.split_data_for_clients_uniformly(all_idx, num_clients = n_folds,
                                                                                 seed = seed,
                                                                                 labels = all_labels_int, keep_labels_proportion = keep_labels_proportion
                                                                                 )

    for i in range(n_folds) :
        print(f"Processing fold {i + 1} / {n_folds}")

        # Create the folder for the current fold
        path_to_save_fold = f'{path_to_save}fold_{i}/'
        os.makedirs(path_to_save_fold, exist_ok = True)

        # Get validation indices for the current fold
        idx_val = idx_per_fold[i]

        # Get training indices for the current fold (all the other indices)
        idx_train = np.concatenate([idx_per_fold[j] for j in range(n_folds) if j != i])

        # Split the training data between clients and save the indices files (both training and validation)
        split_and_save_indices(idx_train, idx_val, all_labels_int[idx_train], path_to_save_fold, print_info = False)
else :
    # Split the data between training and validation
    idx_list_train_val, _ = support_federated_generic.split_data_for_clients(all_idx, percentage_split_per_client = [percentage_data_used_for_training, 1 - percentage_data_used_for_training],
                                                                             seed = seed, labels = all_labels_int, keep_labels_proportion = keep_labels_proportion
                                                                             )

    # Get training and validation indices
    idx_train = idx_list_train_val[0]
    idx_val = idx_list_train_val[1]

    # Split the training data between clients and save the indices files (both training and validation)
    split_and_save_indices(idx_train, idx_val, all_labels_int[idx_train], path_to_save, print_info = True)
    

print("Data splitting and saving of indices files completed.")
