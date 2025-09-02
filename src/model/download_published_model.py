"""
This files contains the functions used to download models published on the PyTorch hub
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
from torchvision import transforms

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_Inception_v3(pretrained = True) :
    """
    Download the inception v3 model
    https://pytorch.org/hub/pytorch_vision_inception_v3/
    """

    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained = pretrained)
    preprocess_functions = get_preprocess_functions('inception')

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
    preprocess_functions = get_preprocess_functions('vgg')
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

def get_preprocess_functions(model_name : str, mean = None, std = None) :
    """
    Given the name of the model return the functions requires to preprocess the input.
    All the functions are inside a torchvision.transforms.Compose() object from torchvision and are functions of the torchvision.transforms package
    Note that the values for normalization (mean and std) are computed on the ImageNet dataset, used originally to train the models.
    If your domain is different, you may need to compute again those values based on your dataset.
    """

    model_name = model_name.lower()

    if model_name == 'inception' :
        if mean is None : mean = [0.485, 0.456, 0.406]
        if std is None  : std  = [0.229, 0.224, 0.225]

        preprocess_functions = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.Normalize(mean = mean, std = std),
            # transforms.ToTensor(),
        ])
    elif model_name == 'vgg' :
        if mean is None : mean = [0.485, 0.456, 0.406]
        if std is None  : std  = [0.229, 0.224, 0.225]

        preprocess_functions  = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean = mean, std = std),
            # transforms.ToTensor(),
        ])
    else :
        raise ValueError(f'model_name not valid. Valid values are inception, vgg. Current value is {model_name}')

    return preprocess_functions


