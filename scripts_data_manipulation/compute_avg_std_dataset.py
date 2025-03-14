"""
Given the path of a dataset compute the avg and std for that dataset
Note that the dataset must contains images


Links with some details/discussion about it : 
https://datascience.stackexchange.com/questions/77084/how-imagenet-mean-and-std-derived
https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import torchvision
import torch

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# path_dataset = './data/ADNI_MRI_2D_Axial_png/'
path_dataset = './data/Kaggle_Alzheimer_MRI_4_classes_dataset'

mode_image_load = torchvision.io.image.ImageReadMode.RGB

n_elements_to_use = -1

preprocess_functions  = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_final_mean_and_std_from_list(mean_list, std_list) :
    # Compute the mean and std for the dataset (for each channel separately)
    mean_dataset = torch.stack(mean_list).mean(dim = 0)
    std_dataset = torch.stack(std_list).mean(dim = 0)

    print(f'Mean dataset : {mean_dataset}')
    print(f'Std dataset  : {std_dataset}')

    return mean_dataset, std_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

list_files_path = support_dataset.get_all_files_from_path(path_dataset)

idx_to_use = np.arange(len(list_files_path))
if n_elements_to_use == -1 :
    idx_to_use = np.arange(len(list_files_path))
elif n_elements_to_use > len(list_files_path) : 
    print(f'Warning: n_elements_to_use ({n_elements_to_use}) is greater than the number of elements in the dataset ({len(list_files_path)}).')
    print(f'Setting n_elements_to_use = {len(list_files_path)}')
    n_elements_to_use = len(list_files_path)
    idx_to_use = np.arange(len(list_files_path))
else :
    idx_to_use = np.random.choice(len(list_files_path), n_elements_to_use, replace = False)

mean_list = []
std_list = []
n_elements_to_print = int(len(idx_to_use) / 20)

# Compute the mean and std for each image
for i in range(len(idx_to_use)) :
    if i % n_elements_to_print == 0 and i != 0 : 
        print(f'Processing {i} / {len(idx_to_use)} ({round(i / len(idx_to_use) * 100, 2)}%)')
        _, _ = compute_final_mean_and_std_from_list(mean_list, std_list)
    
    # Load image
    file_path = list_files_path[idx_to_use[i]]
    image = torchvision.io.read_image(file_path, mode = mode_image_load)
    image = image / 255

    if preprocess_functions is not None :
        image = preprocess_functions (image)
    
    mean_list.append(image.mean(dim = (1, 2)))
    std_list.append(image.std(dim = (1, 2)))


_, _ = compute_final_mean_and_std_from_list(mean_list, std_list)
