"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

try :
    import pydicom as dicom
except ImportError :
    print("The pydicom library is not installed. The MRI_2D_dataset_dicom class will not work.")

from . import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class MRI_2D_dataset(torch.utils.data.Dataset):

    def __init__(self, path_list : list, label_list : list, load_data_in_memory : bool = False, preprocess_functions = None, grey_scale_image : bool = True) :
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

        if len(path_list) != len(label_list) :
            raise ValueError("Length of path_list and label_list must be the same. Current length of path_list : {}, current length of label_list : {}".format(len(path_list), len(label_list)))

        self.path_list = np.asarray(path_list)
        self.labels = torch.asarray(label_list) 

        self.preprocess_functions = preprocess_functions
        self.apply_preprocess_functions = True if preprocess_functions is not None else False
        self.grey_scale_image = grey_scale_image

        if load_data_in_memory : 
            self.load_dataset()
            self.load_data_in_memory = True
        else :
            self.load_data_in_memory = False
            
    def __getitem__(self, idx) :
        if self.load_data_in_memory :
            image = self.data_loaded[idx] 
        else : 
            image = self.load_image(self.path_list[idx], self.grey_scale_image)

        return image, self.labels[idx]
    
    def __len__(self) -> int :
        return len(self.labels)

    def load_image(self, path, create_copy_for_depth : bool = True) :
        """
        Given a path load the image through torchvision.io.read_image
        """

        if type(path) is np.ndarray :
            # In this case someone use a slice idx to get multiple item

            tmp_list = []
            for i in range(len(path)) :
                tmp_list.append(self.__load_single_image(path[i], create_copy_for_depth))

            image = torch.stack(tmp_list)
        else :
            # In this case someone use int idx to get a single item
            image = self.__load_single_image(path, create_copy_for_depth)
            image = image.unsqueeze(0)

        return image


    def __load_single_image(self, path : str, create_copy_for_depth : bool = True) : 
        image = torchvision.io.read_image(path).float()
        image = image / 255.0

        if create_copy_for_depth and self.grey_scale_image : image = self.create_copy_of_image_for_depth_map(image)
        if self.preprocess_functions is not None and self.apply_preprocess_functions : image = self.preprocess_functions(image)

        return image

    def load_dataset(self, create_copy_for_depth : bool = True) :
        """
        Loaded the data inside the memory. This is useful if the dataset is small and can be loaded in memory.
        The data loaded are the images specified in the path_list attribute (self.path_list, passed in the constructor).
        """

        tmp_list = []
        for path in self.path_list : tmp_list.append(self.__load_single_image(path, create_copy_for_depth))
        
        self.data_loaded = torch.stack(tmp_list)
        self.load_data_in_memory = True

    def remove_dataset_from_memory(self) :
        self.data_loaded = None
        self.load_data_in_memory = False
        

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
        image = image.squeeze().permute(1, 2, 0)
    
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
                image_original  = self.load_image(self.path_list[idx])
                print(image_original.mean(), image_original.std())

                # Get data with the preprocess functions
                self.apply_preprocess_functions = True
                image_preprocess = self.load_image(self.path_list[idx])
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

class MRI_2D_dataset_dicom(MRI_2D_dataset):

    def __init__(self, path_list : list, label_list : list, load_data_in_memory : bool = True, preprocess_functions = None) :
        """
        This class work exaclty as MRI_2D_dataset but it expects to receive in input a list with dcm file.
        The only function that change behavior is load_image since it has to load dcm image and convert them in pytorch tensor
        """

        super().__init__(path_list, label_list, load_data_in_memory, preprocess_functions, grey_scale_image = True)

    def load_image(self, path : str, create_copy_for_depth : bool = True) :
        """
        Given a path load a dcm image and convert it to a pytorch tensor.
        """

        image = dicom.dcmread(path).pixel_array
        image[image > 255] = 255 # In some I notidced that there are pixel with value > 255. For now I just set them to 255. In the future I will check if this is the right thing to do.
        image = torch.tensor(image).float().unsqueeze(0)
        image = image / 255.0

        if create_copy_for_depth : image = self.create_copy_of_image_for_depth_map(image)
        if self.preprocess_functions is not None and self.apply_preprocess_functions : image = self.preprocess_functions(image)

        return image
