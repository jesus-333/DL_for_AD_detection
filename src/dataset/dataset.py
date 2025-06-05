"""
Implementations of dataset classes that require png (or images) as inputs.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MRI_dataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset of MRI images.
    The input is a numpy array/torch tensor of shape N x D x H x W, where N is the number of images, D is the depth dimension, H is the height and W is the width.
    Since the data are passed as a numpy array/torch tensor, this class can bu used both for 2D and 3D images.

    Parameters
    ----------
    TODO

    Attributes
    ----------
    TODO
    """

    def __init__(self, data, labels, preprocess_functions = None, print_var = True) :
        # Check input
        if data.shape[0] != len(labels) :
            raise ValueError(f"N. of sampels and n. of labels must be the same. Current n. of samples : {data.shape[0]}, current n. of labels : {len(labels)}")
        
        self.data = preprocess_functions(torch.asarray(data)) if preprocess_functions is not None else torch.asarray(data)
        if print_var : print("DATA LOADED")

        self.labels = torch.asarray(labels)
        if print_var : print("LABELS LOADED")
            
    def __getitem__(self, idx) :
        return self.data[idx], self.labels[idx]
    
    def __len__(self) -> int :
        return len(self.labels)
        
    def visualize_sample(self, idx : int) :
        """
        Visualize a single sample of the dataset
        """

        if self.data.shape[1] >= 3 :
            raise ValueError("This method works only for 2D images, If you use the class for 3D images, use the check_single_sample_V1 or check_single_sample_V2 method.")
        
        # Get the image and the label
        image, label = self.__getitem__(idx)

        # Rearrange the image to have the channel as the last dimension
        image = image.squeeze().permute(1, 2, 0)
    
        # Plot the image
        fig, ax = plt.subplots()

        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'Label : {label}')

        fig.tight_layout()
        fig.show()

    def check_single_sample_V1(self, idx : int, figsize : tuple = None) :
        """
        Plot a single sample of the dataset. This is used to check graphically if the depth map order is correct.
        Note that the depth map should be sorted when the sample is loaded.
        """

        if self.data.shape[1] <= 3 :
            raise ValueError("This method works only for 3D images, i.e. collection of slice. If you use the class for 2D images, use the visualize_sample method.")
    
        # Get image and ensure that the image is a 3D image (i.e. no batch dimension)
        image = self.load_sample(idx).squeeze()
    
        # Get the number of images
        n_depth_map = image.shape[0]

        # Compute the number of rows and columns for the plot
        n_row = int(np.ceil(np.sqrt(n_depth_map)))
        n_col = int(np.ceil(n_depth_map / n_row))
    
        # Create the figure and the axes
        if figsize is None : figsize = (n_col * 3, n_row * 3)
        fig, axs = plt.subplots(n_row, n_col, figsize = figsize)

        for i in range(n_depth_map) :
            # Get the image
            img = image[i].squeeze()
        
            # Plot the image
            axs[i // n_col, i % n_col].imshow(img, cmap = 'gray')
            axs[i // n_col, i % n_col].set_title(f"Depth map order : {i}")

        # Remove the empty subplots
        for i in range(n_depth_map, n_row * n_col) :
            fig.delaxes(axs[i // n_col, i % n_col])
    
        # Show the plot
        fig.tight_layout()
        fig.show()

    def check_single_sample_V2(self, idx : int, figsize : tuple = None) :
        """
        Similar to V1 but it visualize a single image that can be update with a slider.
        """

        if self.data.shape[1] <= 3 :
            raise ValueError("This method works only for 3D images, i.e. collection of slice. If you use the class for 2D images, use the visualize_sample method.")

        # Get image and ensure that the image is a 3D image (i.e. no batch dimension)
        image = self.load_sample(idx).squeeze()
    
        # Get the number of images
        n_depth_map = image.shape[0]

        # Create the figure and the axes
        if figsize is None : figsize = (10, 10)
        fig, ax = plt.subplots(figsize = figsize)

        # Create the slider
        ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])
        slider = plt.Slider(ax_slider, 'Depth map', 0, n_depth_map - 1, valinit = 0, valstep = 1)

        ax.imshow(image[0], cmap = 'gray')
        ax.set_title(f"Depth map order : {0}")

        # Function to update the plot when the slider is moved
        def update(val) :
            idx = int(slider.val)
            img = image[idx].squeeze()
            ax.imshow(img, cmap = 'gray')
            ax.set_title(f"Depth map order : {idx}")
            fig.canvas.draw_idle()

        # Connect the slider to the update function
        slider.on_changed(update)

        # Show the plot
        fig.tight_layout()
        fig.show()

    def move_data_and_labels_to_device(self, device : torch.device) :
        """
        Move the data and labels to the specified device.
        """

        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
