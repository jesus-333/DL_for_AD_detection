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
    data : numpy.ndarray or torch.Tensor
        The data to be used as input. It can be a numpy array or a torch tensor of shape N x D x H x W.
    labels : list or numpy.ndarray
        The labels of the data. It can be a list or a numpy array of shape N.
    preprocess_functions : callable, optional
        A function or a list of functions to preprocess the data. If None, no preprocessing is applied. Default is None. They should be Torchvision transforms or similar.
        Note that the preprocessing functions are applied to all the data during the inizialization of the class. This is done to avoid applying the same preprocessing to each sample during the training.
    print_var : bool, optional
        If True, print additional information when the class is used. Default is True.

    Attributes
    ----------
    data : torch.Tensor
        The data as a torch tensor.
    labels : torch.Tensor
        The labels as a torch tensor.
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

    def move_data_and_labels_to_device(self, device : torch.device) :
        """
        Move the data and labels to the specified device.
        """

        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
        
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
        image, label = self.__getitem__(idx)
        image = image.squeeze()
    
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
        image, label = self.__getitem__(idx)
        image = image.squeeze()
    
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


class MRI_dataset_mmap(MRI_dataset) :
    """
    This class works exactly like the MRI_dataset class, but it is designed to work with memory-mapped files.
    Note that the class is a subclass of MRI_dataset, so it inherits all the methods and attributes of the parent class, apart from the __init__, the __getitem__ and the move_data_and_labels_to_device methods.

    Parameters
    ----------
    path_to_tensor_file : str
        The path to the tensor file that contains the data. The file should be created using torch.save().
    labels : list or numpy.ndarray
        The labels of the data. It can be a list or a numpy array of shape N.
    preprocess_functions : callable, optional
        A function or a list of functions to preprocess the data. If None, no preprocessing is applied. Default is None. They should be Torchvision transforms or similar.
        It should be noted that, unlike the MRI_dataset class, the preprocessing functions are applied to each sample when it is retrieved from the dataset, not during the initialization of the class.
    print_var : bool, optional
        If True, print additional information when the class is used. Default is True.
    """

    def __init__(self, path_to_tensor_file : str, labels, preprocess_functions = None, print_var = True) :
        self.data   = torch.load(f'{path_to_tensor_file}', mmap = True)
        self.labels = torch.asarray(labels)
        self.preprocess_functions = preprocess_functions

        self.path_to_tensor_file = path_to_tensor_file

    def __getitem__(self, idx) :
        return self.preprocess_functions(self.data[idx]) if self.preprocess_functions is not None else self.data[idx], self.labels[idx]

    def move_data_and_labels_to_device(self, device : torch.device) :
        """
        Move the data and labels to the specified device.
        """

        self.data = torch.load(f'{self.path_to_tensor_file}', mmap = True, map_location = device)
        self.labels = self.labels.to(device)
