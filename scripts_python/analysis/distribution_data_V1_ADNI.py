"""
Compare the distribution of the data inside the ADNI dataset.
The script works with the output of `ADNI_extract_middle_slices` (or similar) script inside the `data_manipulation` folder.
Basically it expect to have a single tensor file with all the data stored inside.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

commond_path = "./data/ADNI_axial_middle_slice/"
path_data = f"{commond_path}/dataset_tensor___176_resize.pt"
load_data_in_memory = False

# idx_path = "../../data/ADNI_axial_middle_slice/CENT_idx_517322939"
idx_path = f"{commond_path}/CENT_idx_517322939/"
idx_train_path = f"{idx_path}train_idx_all.npy"
idx_val_path = f"{idx_path}val_idx.npy"

remove_black_pixel = True # If True use a simple filtering (based on pixel value) to remove most of the black pixel (background)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(
        mean = torch.load(f"{commond_path}dataset_mean.pt"),
        std = torch.load(f"{commond_path}dataset_mean.pt")),
])
transforms = None

bins = 200
density = True

x_lim_mean = [0, 0.25]
x_lim_std = [0, 0.25]
# x_lim_mean = None
# x_lim_std = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute mean and std for the dataset

x = torch.load(path_data, mmap = not load_data_in_memory)
if transforms is not None : x = transforms(x)

idx_train = np.load(idx_train_path)
idx_val = np.load(idx_val_path)
x_train = x[idx_train]
x_val = x[idx_val]

def compute_mean_and_std(x, remove_black_pixel : bool) :
    mean_list = np.zeros(len(x))
    std_list = np.zeros(len(x))

    for i in range(len(x)) :
        if remove_black_pixel : tmp_x = x[i][x[i] > 0.02]
        else : tmp_x = x[i]

        mean_list[i] = tmp_x.mean()
        std_list[i] = tmp_x.std()

    return mean_list, std_list

mean_train, std_train = compute_mean_and_std(x_train, remove_black_pixel)
mean_val, std_val = compute_mean_and_std(x_val, remove_black_pixel)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot data

fig, axs = plt.subplots(2, 2)

# Plot the distribution of the mean and std for the train and val set
axs[0, 0].hist(mean_train, bins = bins, density = density)
axs[0, 1].hist(std_train, bins = bins, density = density)
axs[1, 0].hist(mean_val, bins = bins, density = density)
axs[1, 1].hist(std_val, bins = bins, density = density)

# Set the title for each subplot
axs[0, 0].set_title("Mean - Train set")
axs[0, 1].set_title("Std - Train set")
axs[1, 0].set_title("Mean - Val set")
axs[1, 1].set_title("Std - Val set")

for ax in axs.flatten() : ax.grid(True)

if x_lim_mean is not None :
    axs[0, 0].set_xlim(x_lim_mean)
    axs[1, 0].set_xlim(x_lim_mean)

if x_lim_std is not None :
    axs[0, 1].set_xlim(x_lim_std)
    axs[1, 1].set_xlim(x_lim_std)

fig.tight_layout()
fig.show()
