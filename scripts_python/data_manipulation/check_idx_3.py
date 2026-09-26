"""
Check that the split obtained by the script create_idx_files_for_federated_simulations_3.py is valid.

The script produce a "partially leaked" split. The data are divided in two blocks.
One block is randomly split between training and test without considering the original subjects. This constitutes the "leaked" part
The other block is split between training and test (and the training part between clients) while considering the original subjects. I.e. the data of the same subject are always in the same set. They could not appear in both training and test or in two different clients. This constitutes the "non-leaked" part.

What this script does is to ensure that the subject that appear both in the trainind and test set are only those that are in the "leaked" part. 
The subjects that are in the "non-leaked" part should only appear in one of the sets (training or test) and not in both.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (OPTIONAL) CLI Arguments and check

import argparse


parser = argparse.ArgumentParser(description = 'Check that the split obtained by the script create_idx_files_for_federated_simulations_3.py is valid.')

# TODO

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pandas as pd

from addl.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_split_indices_files = "data/ADNI_MRI_Normalized_middle_slice/FL_indices_V3/"
path_dataset_info = "data/ADNI_MRI_Normalized_middle_slice/dataset_info.csv"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

# Load dataset info
dataset_info = pd.read_csv(path_dataset_info)
subj_list_per_sample = dataset_info['subj_id'].to_numpy()

# Get all the indices files
list_idx_files = support_dataset.get_all_files_from_path(path_to_explore = path_split_indices_files, filetype_filter = "npy")

# Get number of clients counting how many files "train_idx_client_i_random_split.npy" there are in the list
# Could also be done with the file "train_idx_client_i_sa_split.npy"
# The script for splitting the data into clients creates three files for each client.
# One for the random split, one for the subject-aware split and one with the two splits combined
n_clients = len([f for f in list_idx_files if "train_idx_client_" in f and "_random_split.npy" in f])

# Load indices for each client
idx_per_client_train_both_splits  = []
idx_per_client_train_sa_split     = [] # sa = subject-aware
idx_per_client_train_random_split = []
for i in range(n_clients):
    idx_per_client_train_both_splits.append(np.load(path_split_indices_files + "train_idx_client_" + str(i) + ".npy"))
    idx_per_client_train_sa_split.append(np.load(path_split_indices_files + "train_idx_client_" + str(i) + "_sa_split.npy"))
    idx_per_client_train_random_split.append(np.load(path_split_indices_files + "train_idx_client_" + str(i) + "_random_split.npy"))

# Load validation indices
idx_array_val_both_split = np.load(path_split_indices_files + "val_idx.npy")
idx_array_val_sa_split = np.load(path_split_indices_files + "val_idx_sa_split.npy")
idx_array_val_random_split = np.load(path_split_indices_files + "val_idx_random_split.npy")

# Load list of subjects used originally for random and subject-aware splits
subj_list_random_split = np.load(path_split_indices_files + "subj_for_random_split.npy")
subj_list_sa_split = np.load(path_split_indices_files + "subj_for_sa_split.npy")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check the training indices for each client

subj_val = set(subj_list_per_sample[idx_array_val_both_split])
subj_train_per_client = [set(subj_list_per_sample[idx]) for idx in idx_per_client_train_both_splits]
subj_train_all = set().union(*subj_train_per_client)
sa_set, rnd_set = set(subj_list_sa_split), set(subj_list_random_split)

# Every subject must belong to exactly one block
assert sa_set.isdisjoint(rnd_set) # This ensures that the two blocks are disjoint
assert (subj_train_all | subj_val) <= (sa_set | rnd_set) # This ensures that all the subjects in the training and validation are a subset of the subjects in the two blocks

# Train/val overlap is only allowed for random-split subjects
overlap = subj_train_all & subj_val
if not overlap <= rnd_set:
    raise ValueError(f"SA subjects in both train and val: {overlap - rnd_set}")

# SA subjects must not span multiple clients
for i in range(n_clients):
    for j in range(i + 1, n_clients):
        shared_sa = subj_train_per_client[i] & subj_train_per_client[j] & sa_set
        if shared_sa:
            raise ValueError(f"SA subjects shared by clients {i} and {j}: {shared_sa}")

# Consistency between the saved sub-splits and their declared blocks
for i in range(n_clients):
    assert set(subj_list_per_sample[idx_per_client_train_sa_split[i]]) <= sa_set
    assert set(subj_list_per_sample[idx_per_client_train_random_split[i]]) <= rnd_set
assert set(subj_list_per_sample[idx_array_val_sa_split]) <= sa_set
assert set(subj_list_per_sample[idx_array_val_random_split]) <= rnd_set

