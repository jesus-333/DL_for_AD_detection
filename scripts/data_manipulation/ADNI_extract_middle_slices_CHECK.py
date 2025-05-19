"""
Used to check the outputs of the ADNI_extract_middle_slice.py script
More precisely check if the information in the data_info.csv correspond to the correct sample in the dataset_tensor.
To do this the script loads a random sample (obtained using conversion_type = 0) and at the same time it load the entire dataset tensor (obtained using conversion_type = 1).
After that it seach the index of the random sample in the dataset tensor. With the index it can get the information from the data_info.csv file using the iloc method.

In this way I have obtained the position of a random sample in the dataset and get the corresponding information stored in the data_info.csv file for that specific sample.
The information are then compared with the information stored in the data_info.json file (that are retrieved instead using the name of the random file).

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import numpy as np
import os
import pandas as pd
import torch
import torchvision

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

path_data = "./data/ADNI_axial_middle_slice/"
path_dataset_info_csv = "./data/ADNI_axial_middle_slice/data_info.csv"
path_dataset_info_json = "./data/ADNI_axial_middle_slice/data_info.json"

create_png_backup = True # If True for each slice saved it creates also a copy in png

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data and create preprocess function

# Get files 
list_of_files = support_dataset.get_all_files_from_path(path_data, filetype_filter = 'pth')

# Get dataset (single tensor)
dataset_tensor = torch.load(f'{path_data}dataset_tensor___176_resize.pt')

# Get data info
data_info_csv = pd.read_csv(path_dataset_info_csv)
with open(path_dataset_info_json) as f: data_info_json = json.load(f)

# Get random file
random_file_path = np.random.choice(list_of_files)
random_file_name = os.path.basename(random_file_path)
random_file_info = data_info_json[random_file_name]
random_sample = torch.load(random_file_path)

# Get idx of the random sample in the dataset
tmp_diff = (dataset_tensor - random_sample).sum(dim = (1, 2, 3))
idx = torch.where(tmp_diff == 0)[0].item()

# Get the info from the csv file
info_csv = data_info_csv.iloc[idx].to_dict()
info_csv['label_int'] = int(info_csv['labels_int'])
info_csv['label_str'] = str(info_csv['labels_str'])

# Compare the info from the csv file and the json file
for key in random_file_info.keys() : print(f"{key} :\n\t{random_file_info[key]} (json)\n\t{info_csv[key]} (csv)")
