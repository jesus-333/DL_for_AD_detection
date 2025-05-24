import torch
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import support_dataset, dataset
from src.model import download_published_model 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_AD = 'data/ADNI_3_fMRI/AD'
path_MCI = 'data/ADNI_3_fMRI/MCI'
path_CN = 'data/ADNI_3_fMRI/CN'

convert_MCI_to_AD = False
model_name = 'vgg'

check_files = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the paths of the files for each class
file_path_list_CN = support_dataset.get_all_files_from_path(path_CN, filetype_filter = 'dcm')
file_path_list_AD = support_dataset.get_all_files_from_path(path_AD, filetype_filter = 'dcm')
file_path_list_MCI = support_dataset.get_all_files_from_path(path_MCI, filetype_filter = 'dcm')

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
preprocess_functions = download_published_model.get_preprocess_functions(model_name)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Create the dataset
fMRI_dataset = dataset.MRI_2D_dataset_dicom(file_path_list, label_list, load_data_in_memory = False, preprocess_functions = preprocess_functions)

# Get a random index
idx_random = np.random.randint(0, len(fMRI_dataset))
image, label = fMRI_dataset[idx_random]

# Plot the image
fMRI_dataset.visualize_sample(idx_random)

# Visualize effect of preprocess function
fMRI_dataset.visualize_preprocess_effect(idx_random)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Create Dataloader
dataloader = torch.utils.data.DataLoader(fMRI_dataset, batch_size = 4, shuffle = True)

# Get a batch
for i, (images, labels) in enumerate(dataloader) :
    print(images.shape, labels.shape)
    break

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Iterate through all the files to check if they are readable

if check_files :

    list_idx_error = []
    list_error = []
    list_path_error = []

    idx_to_print_advancement = np.linspace(0, len(fMRI_dataset), 100).astype(int)

    for i in range(len(fMRI_dataset)) :
        if i in idx_to_print_advancement : 
            print(f"Elements analyzed: {i}/{len(fMRI_dataset)}")

        try :
            image, label = fMRI_dataset[i]
        except Exception as e:
            print("\tError with path number {}".format(i))
            list_idx_error.append(i)
            list_error.append(e)
            list_path_error.append(fMRI_dataset.data[i])

            # Save list to txt files
            with open('error.txt', 'a') as f:
                for i in range(len(list_idx_error)) :
                    string_to_write = f"Error FOUND - path number : {list_idx_error[i]} - Error message : {list_error[i]}\n"
                    string_to_write += f"Path: {list_path_error[i]}\n"
                f.write(string_to_write)

