"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
import torchvision

from . import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class MRI_2D_dataset(torch.utils.data.Dataset):

    def __init__(self, path_list : list, load_data_in_memory : bool = True, preprocess_functions = None):
        """
        """

        if load_data_in_memory : 
            self.data = self.load_dataset(path_list)
        else :
            self.data = path_list

        self.load_data_in_memory = load_data_in_memory
        self.preprocess_functions = preprocess_functions
            
    def __getitem__(self, idx : int) :
        pass
    
    def __len__(self) -> int :
        pass

    def load_image(self, path : str, create_copy_for_depth : bool = True) :
        """
        Given a path load the image through torchvision.io.read_image
        """

        image = torchvision.io.read_image(path).squeeze()
        if create_copy_for_depth : image = self.create_copy_of_image_for_depth_map(image)

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

        


