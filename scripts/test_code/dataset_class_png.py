import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import support_dataset, dataset
from src.model import download_published_model 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_image_AD = './data/ADNI_1_3YR_3T_above_filtered_by_preprocess/AD/'
path_image_CN = './data/ADNI_1_3YR_3T_above_filtered_by_preprocess/CN/'
path_image_MCI = './data/ADNI_1_3YR_3T_above_filtered_by_preprocess/MCI/'

convert_MCI_to_AD = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the paths of the files for each class
file_path_list_CN = support_dataset.get_all_files_from_path(path_image_CN)
file_path_list_AD = support_dataset.get_all_files_from_path(path_image_AD)
file_path_list_MCI = support_dataset.get_all_files_from_path(path_image_MCI)

# Get the labels for each class
label_list_CN = support_dataset.get_labels_from_path_list(file_path_list_CN)
label_list_AD = support_dataset.get_labels_from_path_list(file_path_list_AD)
label_list_MCI = support_dataset.get_labels_from_path_list(file_path_list_MCI)

# Merge all paths and labels
file_path_list = file_path_list_CN + file_path_list_AD + file_path_list_MCI
label_list = label_list_CN + label_list_AD + label_list_MCI
label_list = support_dataset.convert_label_from_str_to_int(label_list, convert_MCI_to_AD = convert_MCI_to_AD)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Create model
model, preprocess_functions = download_published_model.download_vgg_nets(version = 16, batch_normalization = True, pretrained = True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Create the dataset
MRI_dataset = dataset.MRI_2D_dataset(file_path_list, label_list, load_data_in_memory = False, preprocess_functions = preprocess_functions)

# Get a random index
idx_random = np.random.randint(0, len(MRI_dataset))
image, label = MRI_dataset[idx_random]

# Plot the image
MRI_dataset.visualize_sample(idx_random)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Create Dataloader
dataloader = torch.utils.data.DataLoader(MRI_dataset, batch_size = 4, shuffle = True)

# Get a batch
for i, (images, labels) in enumerate(dataloader) :
    print(images.shape, labels.shape)
    break

