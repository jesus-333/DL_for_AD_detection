"""
Implementation of the ViT model through the timm library.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch

try :
    import timm
    from timm.data import resolve_data_config, create_transform
except ImportError :
    error_message = 'The timm library is not installed. Please install it with "pip install timm" and try again.'
    raise ImportError(error_message)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class timm_vit(torch.nn.Module) :

    def __init__(self, config : dict) :
        super().__init__()

        self.model = timm.create_model(
            config['model_name'],
            pretrained = config['pretrained'],
            num_classes = config['num_classes'],
            img_size = config['input_size'],
            in_chans = config['input_channels'],
            embed_dim = config['embed_dim'],
            depth = config['depth'],
            num_heads = config['num_heads'],
        )

    def forward(self, x) :
        return self.model(x)

def get_vit(config : dict) -> timm_vit :
    """
    Function to get the ViT model.

    Parameters
    ----------
    config : dict
        The model config dictionary. For more details see `update_model_config_vit.py`.

    Returns
    -------
    timm_vit
        The ViT model.
    preprocess_functions : torchvision.transforms.Compose
        Preprocessing functions to apply to the input images before feeding them to the model. Returned only if config['preprocess'] is True.
        If config['preprocess'] is False, None is returned instead.
    """
    
    # Get the model
    model = timm_vit(config)
    
    # Get the preprocess functions if needed
    if config['preprocess'] :
        data_config = resolve_data_config({}, model = model.model)
        preprocess_functions = create_transform(**data_config)
    else :
        preprocess_functions = None

    return model, preprocess_functions
