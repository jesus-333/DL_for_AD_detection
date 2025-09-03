"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

from torchvision import transforms

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_activation(activation_name: dict):
    """
    Receive a string and return the relative activation function in pytorch.
    Implemented for relu, elu, selu, gelu
    """

    if activation_name.lower() == 'relu':
        return torch.nn.ReLU()
    elif activation_name.lower() == 'elu':
        return torch.nn.ELU()
    elif activation_name.lower() == 'selu':
        return torch.nn.SELU()
    elif activation_name.lower() == 'gelu':
        return torch.nn.GELU()
    else:
        error_message = 'The activation must have one of the following string: relu, elu, selu, gelu'
        raise ValueError(error_message)

def count_trainable_parameters(layer):
    n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    return n_parameters


def get_preprocess_functions(config : dict = None) -> transforms.Compose :
    """
    Given a config dictionary this function return a torchvision.transforms.Compose with the following preprocessing functions:

    preprocess_functions = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(center_crop_size),
        transforms.Normalize(mean = mean, std = std),
        transforms.ToTensor(),
    ])

    Note that not all the functions are mandatory. If a key is not present in the config dictionary the corresponding function is not added to the Compose list. 
    Invalid values for the parameters (e.g. negative values for resize_size) will raise a ValueError.
    The order of the functions in the Compose list is the one shown above.

    Note that, at least for now, the function does not support additional parameters for the functions (e.g. interpolation for Resize).

    Parameters
    ----------
    config : dict
        The config dictionary can contain the following keys:
            - mean : list of float
                The mean values used for normalization. The length of the list must be the same as std.
            - std : list of float
                The std values used for normalization. The length of the list must be the same as mean.
            - resize_size : int
                The size used for the Resize function. It must be a positive integer.
            - center_crop_size : int
                The size used for the CenterCrop function. It must be a positive integer.
            - add_ToTensor_transformation : bool
                If True the ToTensor transformation is added at the beginning of the preprocessing functions.
                If not present the default value is considered False.

    Returns
    -------
    preprocess_functions : torchvision.transforms.Compose
        A torchvision.transforms.Compose with the preprocessing functions.
    """

    preprocess_functions_list = []

    if 'resize_size' in config :
        if config['resize_size'] > 0 :
            preprocess_functions_list.append(transforms.Resize(config['resize_size']))
        else :
            raise ValueError(f'The value of resize_size must be > 0. Current value is {config["resize_size"]}')

    if 'center_crop_size' in config :
        if config['center_crop_size'] > 0 :
            preprocess_functions_list.append(transforms.CenterCrop(config['center_crop_size']))
        else :
            raise ValueError(f'The value of center_crop_size must be > 0. Current value is {config["center_crop_size"]}')

    if 'mean' in config and 'std' in config :
        if len(config['mean']) == len(config['std']) :
            preprocess_functions_list.append(transforms.Normalize(mean = config['mean'], std = config['std']))
        else :
            raise ValueError(f'The length of mean and std must be the same. Current length are {len(config["mean"])} and {len(config["std"])}')

    if 'add_ToTensor_transformation' in config :
        if config['add_ToTensor_transformation'] is True :
            preprocess_functions_list.insert(0, transforms.ToTensor())
        else :
            pass

    if len(preprocess_functions_list) == 0 :
        raise ValueError('The config dictionary is empty. Please provide at least one preprocessing function.')

    preprocess_functions = transforms.Compose(preprocess_functions_list)

    return preprocess_functions
