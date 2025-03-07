"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
import torchvision
import matplotlib.pyplot as plt

from . import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class MRI_2D_dataset(torch.utils.data.Dataset):

    def __init__(self, path_list : list, label_list : list, load_data_in_memory : bool = True, preprocess_functions = None, grey_scale_image : bool = True) :
        """
        This class is used to create a dataset of 2D MRI images (it can be also used for fMRI data).
        The input is a list of paths to the images and a list of labels.

        args :
        - path_list : list of str
            List of paths to the images. Each path must be a string that can be used to load the image with torchvision.io.read_image
            The images must be 2D images (black and white images), i.e. they must have only one channel.
        - label_list : list of int
            List of labels
        - load_data_in_memory : bool
            If True the data are loaded in memory, otherwise they are loaded on the fly when an image is requested
        - preprocess_functions : list of functions
            List of functions to apply to the image before returning it. They must be from the torchvision.transforms module. If None (the default value) no preprocessing is applied.
        - grey_scale_image : bool
            If True the image is converted to a 3 channel image, where all the 3 channels are the same. This is useful for the models that require a 3 channel image as input.
            If False it means that the image is already a 3 channel image and no conversion is needed.
        """

        if load_data_in_memory : 
            self.data, = self.load_dataset(path_list)
        else :
            self.data = path_list
            
        self.labels = torch.asarray(label_list) 
        self.load_data_in_memory = load_data_in_memory

        self.preprocess_functions = preprocess_functions
        self.apply_preprocess_functions = True if preprocess_functions is not None else False
        self.grey_scale_image = grey_scale_image
            
    def __getitem__(self, idx : int) :
        if self.load_data_in_memory :
            image = self.data[idx] 
        else : 
            image = self.load_image(self.data[idx], self.grey_scale_image)

        return image, self.labels[idx]
    
    def __len__(self) -> int :
        return len(self.data)

    def load_image(self, path : str, create_copy_for_depth : bool = True) :
        """
        Given a path load the image through torchvision.io.read_image
        """

        image = torchvision.io.read_image(path).float()
        image = image / 255.0

        if create_copy_for_depth : image = self.create_copy_of_image_for_depth_map(image)
        if self.preprocess_functions is not None and self.apply_preprocess_functions : image = self.preprocess_functions(image)

        return image

    def load_dataset(self) :
        pass

    def create_copy_of_image_for_depth_map(self, image) :
        """
        Usually the MRI scan are black and white images, with a single channel (or depth map).
        This function create copy of the input and stack them along the channel dimension.
        """
        
        image = torch.cat([image, image, image])

        return image

    def visualize_sample(self, idx : int) :
        """
        Visualize a single sample of the dataset
        """
        
        # Get the image and the label
        image, label = self.__getitem__(idx)

        # Rearrange the image to have the channel as the last dimension
        image = image.permute(1, 2, 0)
    
        # Plot the image
        fig, ax = plt.subplots()

        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'Label : {label}')

        fig.tight_layout()
        fig.show()       


    def visualize_preprocess_effect(self, idx : int) :
        """
        Visualize a single sample of the dataset, before and after the preprocessing.
        This function works only if the preprocess_functions attribute is not None.
        """

        if self.preprocess_functions is None :
            raise ValueError("The preprocess_functions attribute is None. No preprocessing is applied to the images.")
        else :
            original_setting = self.apply_preprocess_functions

            if self.load_data_in_memory :
                pass
            else :
                # Get data without applying the preprocess functions
                self.apply_preprocess_functions = False
                image_original  = self.load_image(self.data[idx])
                print(image_original.mean(), image_original.std())

                # Get data with the preprocess functions
                self.apply_preprocess_functions = True
                image_preprocess = self.load_image(self.data[idx])
                print(image_preprocess.mean(), image_preprocess.std())

            # Rearrange the images to have the channel as the last dimension
            image_original = image_original.permute(1, 2, 0)
            image_preprocess = image_preprocess.permute(1, 2, 0)
        
            # Plot the image
            fig, axs = plt.subplots(1, 2)
            
            axs[0].imshow(image_original, cmap = 'gray')
            axs[0].set_title('Original Image')

            axs[1].imshow(image_preprocess, cmap = 'gray')
            axs[1].set_title('Preprocessed Image')

            fig.tight_layout()
            fig.show()       


