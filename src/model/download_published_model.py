"""
This files contains the functions used to download models published on the PyTorch hub
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

from . import support_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_Inception_v3(pretrained = True) :
    """
    Download the inception v3 model
    https://pytorch.org/hub/pytorch_vision_inception_v3/
    """

    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = pretrained)
    preprocess_functions_config = get_preprocess_functions_config_for_specific_model('inception')
    preprocess_functions = support_model.get_preprocess_functions(preprocess_functions_config)

    return model, preprocess_functions

def download_resnet50(pretrained = True) :
    """
    Download the resnet50 model
    https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50/
    """
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained = pretrained)

    return model

def download_vgg_nets(version : int, batch_norm : bool, pretrained = True) :
    """
    Download a version of the VGG network
    https://pytorch.org/hub/pytorch_vision_vgg/
    """

    if version != 11 and version != 13 and version != 16 and version != 19 :
        raise ValueError(f"Version is not valid. Possible version are 11, 13, 16 or 19. Current value is {version}")

    version_name = f'vgg{version}'
    if batch_norm : version_name += '_bn'

    model = torch.hub.load('pytorch/vision:v0.10.0', version_name, pretrained = pretrained)
    preprocess_functions_config = get_preprocess_functions_config_for_specific_model('vgg')
    preprocess_functions = support_model.get_preprocess_functions(preprocess_functions_config)
    print("Note that the the input images for this model have to be loaded in to a range of [0, 1]")

    return model, preprocess_functions

def download_resnet_3D(pretrained = True) :
    """
    Get the resnet 3D

    https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
    """

    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained = pretrained)

    return model

def download_UNet(pretrained = True) :
    """
    Get the pretrained UNet, a model for abnormality segmentation on a dataset of brain MRI volumes

    https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
    https://arxiv.org/abs/1505.04597
    """

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels = 3, out_channels = 1, init_features = 32, pretrained = pretrained)

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_preprocess_functions_config_for_specific_model(model_name : str, add_ToTensor_transformation : bool = False) -> dict :
    """
    Given the name of the model return the config required by the function get_preprocess_functions() (inside support_model.py) for that specific model.
    The values are the ones specified in the pytorch hub webpage for the model.

    Parameters
    ----------
    model_name : str
        The name of the model. Current possible values are 'inception', 'vgg'
    add_ToTensor_transformation : bool
        If True the ToTensor transformation is added at the beginning of the preprocessing functions.
        Default value is False.

    Returns
    -------
    config : dict
        A dictionary with the following keys:
            - mean : list of float
                The mean values used for normalization
            - std : list of float
                The std values used for normalization
            - resize_size : int
                The size used for the Resize function
            - center_crop_size : int
                The size used for the CenterCrop function
            - add_ToTensor_transformation : bool
                If this key is present inside the config the functions get_preprocess_functions() will add the ToTensor transformation at the beginning of the preprocessing functions.
                This key is added only if the parameter add_ToTensor_transformation is True.
    """

    model_name = model_name.lower()

    if model_name == 'inception' :
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        resize_size = 299
        center_crop_size = 299

    elif model_name == 'vgg' :
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        resize_size = 256
        center_crop_size = 224
    else :
        implemented_model = ['incpetion', 'vgg']
        raise ValueError(f'model_name not valid. Valid values are {implemented_model}. Current value is {model_name}')

    config = dict(
        mean = mean,
        std = std,
        resize_size = resize_size,
        center_crop_size = center_crop_size,
    )

    if add_ToTensor_transformation : config['add_ToTensor_transformation'] = True

    return config


