"""
Check the idx files produced by `create_idx_files_for_federated_simulations_2.py`

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pandas as pd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_data = "data/ADNI_axial_middle_slice/"
name_tensor_file = "dataset_tensor___176_resize.pt"
dataset_info_file = "dataset_info.csv"

path_indices_files = "data/ADNI_axial_middle_slice/FL_indices_V2/"
path_train_idx = f"{path_indices_files}train_idx_all.npy"
path_val_idx = f"{path_indices_files}val_idx.npy"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load dataset info and idx files

dataset_info = pd.read_csv(f'{path_data}{dataset_info_file}')

idx_train = np.load(path_train_idx)
idx_val = np.load(path_val_idx)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get the list of subjects in the train and val sets

def get_subj(dataset_info, idx_list) :
    subj_list = []
    for i in range(len(idx_list)) :
        idx = idx_list[i]
        subj_list.append(dataset_info.iloc[idx].subj_id)
    return np.unique(subj_list)

subj_train = get_subj(dataset_info, idx_train)
subj_val = get_subj(dataset_info, idx_val)

print(f"Number of subjects in the train set : {len(subj_train)}")
print(f"Number of subjects in the val set : {len(subj_val)}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check if there are subjects in common between the train and val sets

subj_in_common = np.intersect1d(subj_train, subj_val)
print(f"Number of subjects in common between the train and val sets : {len(subj_in_common)}")

