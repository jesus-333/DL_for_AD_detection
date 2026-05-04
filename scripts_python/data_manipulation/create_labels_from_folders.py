"""
When you download imaging data from ADNI, usually you get a folder for each subject, and inside the folder there are the images of that subject, divided in different folders (usually one for each visit).
This script takes as input a folder, get all the files with a specific filetype (e.g. .nii, .dcm) and for each file it pairs the subject id with the label (both in int and str format).
The result is saved in a csv file and json file.

Note that the labels for all subjects of ADNI were predownloaded and saved separately.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import numpy as np
import pandas as pd

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_data = "./data/ADNI_MRI_Normalized/"
extension_to_filter = 'nii'


labels_file_int = "./data/ADNI_Labels/all_labels_int.json"
labels_file_str = "./data/ADNI_Labels/all_labels_str.json"

n_element_to_print = 200

label_to_int = dict(
    CN    = 0,
    AD    = 1,
    MCI   = 2,
    EMCI  = 3,
    LMCI  = 4,
    SMC   = 5,
    Patient = 6,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get files and labels

# Get files 
list_of_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = extension_to_filter)

# Read the labels
all_labels_int = json.load(open(labels_file_int))
all_labels_str = json.load(open(labels_file_str))

# Variable used to count the number of subjects for each label in the dataset
count_subj_per_label = dict()
for label in label_to_int : count_subj_per_label[label] = 0

# Variable used to create the csv file
labels_int_dataset = []
labels_str_dataset = []
subj_id_dataset = []

# Variable used to create the json file
dict_dataset_info = dict()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get labels for each file in the dataset

for i in range(len(list_of_files)) :
    if i % n_element_to_print == 0 : print(f"Processing file {i}/{len(list_of_files)}\t({round(i / len(list_of_files) * 100, 2)}%)")

    # Get file path
    file_path = list_of_files[i]

    # Create file path for the image
    file_path_decomposition = file_path.split('/')
    
    # Get info
    file_name = file_path_decomposition[-1]
    subj_id = file_path_decomposition[-5]
    date_aquisition = file_path_decomposition[-3]
    tmp_id = file_path_decomposition[-2]

    # Get label for the subject
    label_int = all_labels_int[subj_id]
    label_str = all_labels_str[subj_id]

    # Save labels
    labels_int_dataset.append(label_int)
    labels_str_dataset.append(label_str)

    # Save subject id
    subj_id_dataset.append(subj_id)

    # Save info in the dict for the json file
    dict_dataset_info[file_name] = {
        'subj_id' : subj_id,
        'label_int' : label_int,
        'label_str' : label_str,
    }

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create and save dataset info

# Create dataframe with dataset info
df_dataset = pd.DataFrame(data = None, columns = ['subj_id', 'labels_int', 'labels_str', 'file_name'])
df_dataset['subj_id'] = subj_id_dataset
df_dataset['labels_int'] = labels_int_dataset
df_dataset['labels_str'] = labels_str_dataset
df_dataset['file_name'] = list_of_files

# Save dataset info (csv)
df_dataset.to_csv(f'{path_data}data_info.csv', index = False)

# Save dataset info (json)
with open(f'{path_data}data_info.json', 'w') as fp: json.dump(dict_dataset_info, fp)


