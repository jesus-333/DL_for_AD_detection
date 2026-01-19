"""
Measure how long it takes to read the entire fMRI dataset and iterate through it through torch dataloader.
By specifieing the model_name, the preprocess functions for the specific model will be also used during data loading.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import time
import numpy as np

from src.dataset import dataset_png, support_dataset
from src.model import download_published_model, support_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
path_AD = 'data/ADNI_3_fMRI/AD'
path_MCI = 'data/ADNI_3_fMRI/MCI'
path_CN = 'data/ADNI_3_fMRI/CN'

convert_MCI_to_AD = False
model_name = 'vgg'

batch_size = 100
n_repetition = 2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get the paths of the files for each class
file_path_list_CN = support_dataset.get_all_files_from_path(path_CN)
file_path_list_AD = support_dataset.get_all_files_from_path(path_AD)
file_path_list_MCI = support_dataset.get_all_files_from_path(path_MCI)

# Get the labels for each class
label_list_CN = support_dataset.get_labels_from_path_list(file_path_list_CN)
label_list_AD = support_dataset.get_labels_from_path_list(file_path_list_AD)
label_list_MCI = support_dataset.get_labels_from_path_list(file_path_list_MCI)

# Merge all paths and labels
file_path_list = file_path_list_CN + file_path_list_AD + file_path_list_MCI
label_list = label_list_CN + label_list_AD + label_list_MCI
label_list = support_dataset.convert_label_from_str_to_int(label_list, convert_MCI_to_AD = convert_MCI_to_AD)

print("File path readed")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create model
if model_name is not None :
    preprocess_functions_config = download_published_model.get_preprocess_functions_config_for_specific_model(model_name)
    preprocess_functions = support_model.get_preprocess_functions(preprocess_functions_config)
    print('prepocess functions loaded')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create the dataset
fMRI_dataset = dataset_png.MRI_2D_dataset_dicom(file_path_list, label_list, load_data_in_memory = False, preprocess_functions = preprocess_functions)
print('Dataset created')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create Dataloader
dataloader = torch.utils.data.DataLoader(fMRI_dataset, batch_size = 30, shuffle = True)

list_time_dataloader = []
for i in range(n_repetition) :
    print('Repetition {}/{}'.format(i + 1, n_repetition))

    start = time.time()
    for j, (images, labels) in enumerate(dataloader) :
        if j % batch_size == 0 :
            print('\tBatch {}/{}'.format(j, len(dataloader)))

    end = time.time()
    list_time_dataloader.append(end - start)
    print('Repetition {}/{}'.format(i + 1, n_repetition))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Print results
print('{:.2f}±{:.2f}ms'.format(np.mean(list_time_dataloader) * 1000, np.std(list_time_dataloader) * 1000))
