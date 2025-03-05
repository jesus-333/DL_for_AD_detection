"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
import torchvision
import matplotlib.pyplot as plt

from . import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class MRI_2D_dataset(torch.utils.data.Dataset):

    def __init__(self, path_list : list, label_list : list, load_data_in_memory : bool = True, preprocess_functions = None):
        """
        """

        if load_data_in_memory : 
            self.data, = self.load_dataset(path_list)
        else :
            self.data = path_list
            
        self.labels = torch.asarray(label_list) 
        self.load_data_in_memory = load_data_in_memory
        self.preprocess_functions = preprocess_functions
            
    def __getitem__(self, idx : int) :
        if self.load_data_in_memory :
            image = self.data[idx] 
        else : 
            image = self.load_image(self.data[idx])

        return image, self.labels[idx]
    
    def __len__(self) -> int :
        return len(self.data)

    def load_image(self, path : str, create_copy_for_depth : bool = True) :
        """
        Given a path load the image through torchvision.io.read_image
        """

        image = torchvision.io.read_image(path)
        if create_copy_for_depth : image = self.create_copy_of_image_for_depth_map(image)

        if self.preprocess_functions is not None : image = self.preprocess_functions(image)

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
        Visualize a sample of the dataset
        """
        
        # Get the image and the label
        image, label = self.__getitem__(idx)

        # Rearrange the image to have the channel as the last dimension
        image = image.permute(1, 2, 0)
        print(image.shape)
    
        # Plot the image
        fig, ax = plt.subplots()

        ax.imshow(image, cmap = 'gray')
        ax.set_title(f'Label : {label}')

        fig.tight_layout()
        fig.show()       


