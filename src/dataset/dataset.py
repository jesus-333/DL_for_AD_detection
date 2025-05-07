"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

try :
    import pydicom as dicom
except ImportError :
    print("The pydicom library is not installed. The MRI_2D_dataset_dicom class will not work.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset 2D

class MRI_2D_dataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset of 2D MRI images.
    The input is a list of paths to the images and a list of labels.

    Parameters
    ----------
    - path_list : list of str
        List of paths to the images. Each path must be a string that can be used to load the image with torchvision.io.read_image
    - labels_list : list of int
        List of labels
    - load_data_in_memory : bool
        If True the data are loaded in memory, otherwise they are loaded on the fly when an image is requested
    - preprocess_functions : list of functions
        List of functions to apply to the image before returning it. They must be from the torchvision.transforms module. If None (the default value) no preprocessing is applied.
        They can be set after the creation of the object with the set_preprocess_functions method.
    - grey_scale_image : bool
        If True the images are loaded with ImageReadMode.GRAY. If False the images are loaded with ImageReadMode.RGB.
        Default is False.

    Attributes
    ----------
    - path_list : np.ndarray
        List of paths to the images. Each path must be a string that can be used to load the image with torchvision.io.read_image
    - labels : torch.Tensor
        List of labels. It is a tensor of int. The values are the same as the input labels_list.
    - preprocess_functions : torchvision.transforms.Compose
        List of functions to apply to the image before returning it, if passed in the constructor. If None no preprocessing is applied.
    - apply_preprocess_functions : bool
        If True the preprocess_functions are applied to the images. If False they are not applied. This can be useful if you want to apply the preprocess functions only in some cases and avoid the recreation of the dataset.
    - grey_scale_image : bool
        If True the images are loaded with ImageReadMode.GRAY. If False the images are loaded with ImageReadMode.RGB.
    - add_extra_dimensions_to_single_sample : bool
        If True when a single sample is required, an extra dimension is added to the tensor. By default this is False, to ensure compatibility with the DataLoader class.
        If you use the dataset and you do not required a DataLoader, you can set this attribute to True, so even the shape of single samples is (1, 1, H, W) (and therefore directly compatible with convolutional layers).
    - load_data_in_memory : bool
        This attribute is set automatically by the constructor. It specifies if the data are loaded in memory or not and is later used in the __getitem__ method.
    """

    def __init__(self, path_list : list, labels_list : list, load_data_in_memory : bool = False, preprocess_functions = None, grey_scale_image : bool = False) :
    
        # Check input
        if len(path_list) != len(labels_list) :
            raise ValueError("Length of path_list and labels_list must be the same. Current length of path_list : {}, current length of labels_list : {}".format(len(path_list), len(labels_list)))

        self.path_list = np.asarray(path_list)
        self.labels = torch.asarray(labels_list)

        self.preprocess_functions = preprocess_functions
        self.apply_preprocess_functions = True if preprocess_functions is not None else False
        self.grey_scale_image = grey_scale_image
        self.add_extra_dimensions_to_single_sample = False

        if load_data_in_memory :
            self.load_dataset()
            self.load_data_in_memory = True
        else :
            self.load_data_in_memory = False
            
    def __getitem__(self, idx) :
        """
        Note in case you want to load multiple images when self.load_data_in_memory is False and you not use any preprocess functions (i.e. preprocess_functions is None).
        You must be sure that each image has the same shape. If the images have different shapes an error will occur!
        This is also true if the preprocess functions do not change the shape of the images.

        Also notes that when a single image is requested, the shape of the image is (1, H, W) if add_extra_dimensions_to_single_sample is False (i.e. no batch dimension), otherwise it is (1, 1, H, W).
        """

        if self.load_data_in_memory :
            image = self.data_loaded[idx]
        else :
            image = self.load_image(self.path_list[idx])

        return image, self.labels[idx]
    
    def __len__(self) -> int :
        return len(self.labels)

    def load_image(self, path) :
        """
        Given a path load the image through torchvision.io.read_image

        Note if yout want to load multiple images when self.load_data_in_memory is False and you not use any preprocess functions (i.e. preprocess_functions is None).
        In this case you must be sure that each image has the same shape. If the images have different shapes an error will occur.
        This is also true if the preprocess functions do not change the shape of the images.
        """

        if type(path) is np.ndarray or type(path) is list :
            # In this case someone use a slice idx to get multiple item

            tmp_list = []
            for i in range(len(path)) :
                tmp_list.append(self.load_single_image(path[i]))
            
            image = torch.stack(tmp_list)
        else :
            # In this case someone use int idx to get a single item
            image = self.load_single_image(path)
            if self.add_extra_dimensions_to_single_sample : image = image.unsqueeze(0)

        return image

    def load_single_image(self, path : str) :
        # Load the image
        if self.grey_scale_image :
            image = torchvision.io.read_image(path, mode = torchvision.io.image.ImageReadMode.GRAY).float()
        else :
            image = torchvision.io.read_image(path, mode = torchvision.io.image.ImageReadMode.RGB).float()

        # Normalize in [0, 1] range
        image = image / 255.0

        if self.preprocess_functions is not None and self.apply_preprocess_functions : image = self.preprocess_functions(image)

        return image

    def load_dataset(self) :
        """
        Loaded the data inside the memory. This is useful if the dataset is small and can be loaded in memory.
        The data loaded are the images specified in the path_list attribute (self.path_list, passed in the constructor).
        """

        tmp_list = []
        for path in self.path_list : tmp_list.append(self.load_single_image(path))
        
        self.data_loaded = torch.stack(tmp_list)
        self.load_data_in_memory = True

    def remove_dataset_from_memory(self) :
        self.data_loaded = None
        self.load_data_in_memory = False
        
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

    def compute_avg_and_std_for_dataset(self, n_samples : int = -1) :
        """
        Compute the average and the standard deviation, using n_samples from the dataset.
        If the preprocess_functions attribute is not None, some of them can be applied to the images before computing the average and the standard deviation.
        More precisely, Resize and CenterCrop, if present in the preprocess_functions, are applied.

        Links with some details/discussion about it : 
            https://datascience.stackexchange.com/questions/77084/how-imagenet-mean-and-std-derived
            https://discuss.pytorch.org/t/discussion-why-normalise-according-to-imagenet-mean-and-std-dev-for-transfer-learning/115670

        Parameters
        ----------
        n_samples : int
            Number of samples to use to compute the average and the standard deviation. If -1 (the default value) all the samples are used.
        """

        if self.preprocess_functions is not None :
            # Backup of the current preprocess_functions
            preprocess_functions_backup = self.preprocess_functions
            
            # New preprocess_functions with only Resize and CenterCrop
            self.preprocess_functions = torchvision.transforms.Compose([])

            # Check if Resize and CenterCrop are present in the preprocess_functions
            for transform in preprocess_functions_backup.transforms :
                # If one of the two is present, add it to the new preprocess_functions
                if isinstance(transform, torchvision.transforms.Resize) or isinstance(transform, torchvision.transforms.CenterCrop) :
                    self.preprocess_functions.transforms.append(transform)

        if n_samples == -1 or n_samples >= len(self) :
            images = self[:][0]
        else :
            idx_random = np.random.choice(len(self), n_samples, replace = False)
            images = self[idx_random][0]

        # Compute the average and the standard deviation
        avg = images.mean(dim = (0, 2, 3))
        std = images.std(dim = (0, 2, 3))

        if self.preprocess_functions is not None :
            # Restore the original preprocess_functions
            self.preprocess_functions = preprocess_functions_backup

        return avg, std

    def update_preprocess_functions_normalize(self) :
        """
        Update the normalize transforms, if present in the preprocess_functions.
        The mean and the standard deviation are computed using the compute_avg_and_std_for_dataset method.
        """

        # Check if the preprocess_functions attribute is not None
        if self.preprocess_functions is None :
            print("The preprocess_functions attribute is None. No preprocessing is applied to the images.")
        else :
            # Check if the normalize transform is present in the preprocess_functions
            normalize_present = False
            for transform in self.preprocess_functions.transforms :
                if isinstance(transform, torchvision.transforms.Normalize) :
                    # Compute the average and the standard deviation
                    avg, std = self.compute_avg_and_std_for_dataset()

                    # Update the mean and the standard deviation
                    transform.mean = avg
                    transform.std = std
                    
                    print("preprocess_functions before : ", self.preprocess_functions, "\n")
                    print("The mean and the standard deviation of the normalize transform have been updated.")
                    print("The new mean values are : {}".format(avg))
                    print("The new std values are  : {}".format(std), "\n")
                    print("preprocess_functions after : ", self.preprocess_functions)

                    normalize_present = True

            if not normalize_present : print("The normalize transform is not present in the preprocess_functions attribute.")

    def set_preprocess_functions(self, preprocess_functions) :
        """
        Set the preprocess_functions attribute.
        """

        if isinstance(preprocess_functions, torchvision.transforms.Compose) :
            self.preprocess_functions = preprocess_functions
            self.apply_preprocess_functions = True if preprocess_functions is not None else False
        else :
            raise ValueError("The preprocess_functions must be an instance of torchvision.transforms.Compose")

class MRI_2D_dataset_dicom(MRI_2D_dataset):
    """
    This class work exaclty as MRI_2D_dataset but it expects to receive in input a list with dcm file.
    The only function that change behavior is load_image since it has to load dcm image and convert them in pytorch tensor.
    
    NOT USED FOR NOW.
    """

    def __init__(self, path_list : list, labels_list : list, load_data_in_memory : bool = True, preprocess_functions = None) :

        super().__init__(path_list, labels_list, load_data_in_memory, preprocess_functions, grey_scale_image = True)

    def load_image(self, path : str, create_copy_for_depth : bool = True) :
        """
        Given a path load a dcm image and convert it to a pytorch tensor.
        """
        
        # Read DCM file
        image = dicom.dcmread(path).pixel_array
        image[image > 255] = 255 # In some I notidced that there are pixel with value > 255. For now I just set them to 255. In the future I will check if this is the right thing to do.

        # Convert to tensor
        image = torch.tensor(image).float().unsqueeze(0)

        # Normalize in [0, 1] range
        image = image / 255.0

        if create_copy_for_depth : image = self.create_copy_of_image_for_depth_map(image)
        if self.preprocess_functions is not None and self.apply_preprocess_functions : image = self.preprocess_functions(image)

        return image

    def create_copy_of_image_for_depth_map(self, image) :
        """
        Usually the MRI scan are black and white images, with a single channel (or depth map).
        This function create copy of the input and stack them along the channel dimension, to obtain an "RGB image".
        Note that this is not a real RGB image, since all the channels contain the same values. 
        Nonetheless, this can be useful to use models that require RGB images as input (e.g. VGG, Inception, etc.).
        """
        
        image = torch.cat([image, image, image])

        return image

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset 3D

class MRI_3D_dataset(MRI_2D_dataset) :
    """
    This class is used to create a dataset of 3D MRI images.
    Each sample of the dataset is a single MRI scan, with a single label. The shape of each sample is (1, D, H, W), where D is the depth of the image (number of slices), H is the height and W is the width.
    Note that the D dimension is also referred to as z-dimension in the MRI field.

    The class expects that each image is a greymap png file. The images are loaded using the torchvision.io.read_image function and normalized in the range [0, 1].
    Since the class reads images it is creaated as an extension of the MRI_2D_dataset class. Therefore it inherits the methods load_image, __len__, and load_single_image. The methods __init__, __getitem__, and load_dataset are overridden to adapt them to the 3D images.

    Parameters
    ----------
    - paths_dict : dict
        Dictionary with the paths to the images. The keys are the paths to folder containing the images. The values are lists of paths to the images. An example of the dictionary can be obtained using the get_all_files_from_path_divided_per_folder function inside the support_dataset module.
    - depth_map_order_dict : dict
        Dictionary with the order of the slices for each image. The keys are the paths to folder containing the images. The values are lists of integers, that specify the order of the slices. This is necessary to create the depth map of the image. The order of the slices is important because it determines the order of the slices in the 3D image.
        E.g. Let's assume that in the path_dict and depth_map_order_dict we have a key "folder_1" with the following values: paths_dict["folder_1"] = ["img_2.png", "img_0.png", "img_1.png"]. Then the depth_map_order_dict["folder_1"] must be a list/array with the following values : [1, 2, 0].
             So thhe first element of depth_map_order_dict["folder_1"] contains the position of the first slice in paths_dict["folder_1"] and so on.
    - labels_dict : dict
        Dictionary with the labels for each folder. It must have the same key of paths_dict and depth_map_order_dict.
    - load_data_in_memory : bool
        If True the data are loaded in memory, otherwise they are loaded on the fly when an image is requested. Default is False.
    - load_data_type : int
        If 0 the data are loaded in memory as a list of tensors. If 1 the data are loaded in memory as a single tensor. Default is 0. This parameter is used only if load_data_in_memory is True.
    - preprocess_functions : list of functions
        List of functions to apply to the image before returning it. They must be from the torchvision.transforms module. If None (the default value) no preprocessing is applied.
        They can be set after the creation of the object with the set_preprocess_functions method (the method is inherited from the MRI_2D_dataset class).

    Attributes
    ----------
    TODO

    """

    def __init__(self, paths_dict : dict, depth_map_order_dict : dict, labels_dict : dict, load_data_in_memory : bool = False, load_data_type : int = 0, preprocess_functions = None) :
        # Temporary variable to save informations
        folder_list = []                    # List with all the folders
        files_per_folder = []               # List where each element is a list with the files of the folder. E.g. files_per_folder[0] is a list with the files of the folder folder_list[0]
        depth_map_order_per_folder = []     # List where each element is a list with the depth map order of the folder. E.g. depth_map_order_per_folder[0] is a list with the depth map order of the files inside folder folder_list[0]
        labels_list = []                     # List of the labels

        # Convert key of the dictionary to an array and save the files for each folder
        for (idx, folder) in enumerate(paths_dict.keys()) :
            folder_list.append(folder)
            files_per_folder.append([])
            depth_map_order_per_folder.append([])
            labels_list.append(labels_dict[folder])

            # Check if list of files in the paths_dict and depth_map_order_dict have the same number of elements.
            if len(paths_dict[folder]) != len(depth_map_order_dict[folder]) :
                raise ValueError(f"Length of paths_dict and depth_map_order_dict must be the same for each folder. Folder currently analyzed : {folder}. Current length of paths_dict : {len(paths_dict[folder])}, current length of depth_map_order_dict : {len(depth_map_order_dict[folder])}")
            
            for j in range(len(paths_dict[folder])) :
                # Save the files
                file_path = paths_dict[folder][j]
                files_per_folder[idx].append(file_path)

                # Save the depth map order
                depth_map_order = depth_map_order_dict[folder][j]
                depth_map_order_per_folder[idx].append(depth_map_order)
        
        # Convert the lists to numpy arrays
        self.folder_list = np.asarray(folder_list)
        self.files_per_folder = np.asarray(files_per_folder)
        self.depth_map_order_per_folder = np.asarray(depth_map_order_per_folder)
        self.labels = torch.asarray(labels_list)
    
        # Used in the parent class
        self.grey_scale_image = True
        self.preprocess_functions = preprocess_functions
        self.add_extra_dimensions_to_single_sample = False
        self.apply_preprocess_functions = True if preprocess_functions is not None else False

        # Load the data in memory if requested
        if load_data_in_memory :
            self.load_dataset(load_data_type)
        else :
            self.load_data_in_memory = False
            self.load_data_type = -1

    def __getitem__(self, idx) :
        """
        Get the item at the specified index. The return value is a tuple (image, label).

        Notes that when a single image is requested, the shape of the image is (n_depth_map, H, W) if add_extra_dimensions_to_single_sample is False (i.e. no batch dimension), otherwise it is (1, 1, H, W).
        """
        if self.load_data_in_memory :
            image = self.data_loaded[idx]

            # If the data are loaded as a list of tensors I create a single tensor to return
            # The check of the type is done to avoid the error when I ask a single sample i.e. idx = n, with n an integer
            # In that case self.data_loaded[idx] is a tensor with a shape of (D, H, W) and not a list of tensors
            # Instaead if idx is something like n:m (i.e. a slice) the self.data_loaded[idx] is a list of tensors
            # In that case I stack the tensors to create a single tensor
            if self.load_data_type == 0 and type(image) is not torch.Tensor : image = torch.stack(image)
        else :
            image = self.load_sample(idx)

        return image, self.labels[idx]

    def load_sample(self, idx) :
        """
        Load the sample(s) specified by the idx parameter.
        """

        if type(idx) is np.ndarray or type(idx) is list:
            # In this case someone use a slice idx to get multiple item

            tmp_list = []
            for i in range(len(idx)) :
                tmp_list.append(self.load_singe_sample(idx[i]))
            
            image = torch.stack(tmp_list)
        else :
            # In this case someone use int idx to get a single item
            image = self.load_singe_sample(idx)
            if self.add_extra_dimensions_to_single_sample : image = image.unsqueeze(0)

        return image

    def load_singe_sample(self, idx : int) :
        """
        Load a single sample from the dataset, specified by the idx parameter.
        The order of the depth map is specified by the depth_map_order_per_folder attribute.
        """
    
        files_in_the_current_folder = self.files_per_folder[idx]
        depth_map_order = self.depth_map_order_per_folder[idx]

        single_sample_images = []

        for i in range(len(files_in_the_current_folder)) :
            # Create the path to the image
            path_image = files_in_the_current_folder[depth_map_order[i]]

            # Load the image
            image = self.load_single_image(path_image).squeeze()

            # Add the image to the sample
            single_sample_images.append(image)

        # Stack the images along the depth dimension
        single_sample_images = torch.stack(single_sample_images)

        # Return the image
        return single_sample_images

    def load_dataset(self, load_data_type : int = 0) :
        """
        Loaded the data inside the memory. This is useful if the dataset is small and can be loaded in memory.
        """

        tmp_list = []
        n_element_to_print = int(len(self.labels) / 10)
        for i in range(len(self.folder_list)) :
            if (i + 1) % n_element_to_print == 0 : print(f"Loading sample {i + 1}/{len(self.folder_list)}   \t({round((i + 1) / len(self.folder_list) * 100, 2)}%)")

            sample = self.load_singe_sample(i)
            tmp_list.append(sample)
        
        if load_data_type == 0 :
            # In this case the data are loaded as a list of tensors
            self.data_loaded = tmp_list
        elif load_data_type == 1 :
            # In this case the data are loaded as a single tensor
            self.data_loaded = torch.stack(tmp_list)
        else :
            raise ValueError(f"load_data_type must be 0 (list of tensors) or 1 (single tensor). Current value : {load_data_type}")

        self.load_data_type = load_data_type
        self.load_data_in_memory = True

    def check_single_sample_V1(self, idx : int, figsize : tuple = None) :
        """
        Plot a single sample of the dataset. This is used to check graphically if the depth map order is correct.
        Note that the depth map should be sorted when the sample is loaded.
        """
    
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
