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
        
        # Check the config
        self.required_parameters = ['model_name', 'num_classes']
        self.check_config(config)
        
        # Create kwargs dictionary with the parameters to pass to the timm
        kwargs = {k: v for k, v in config.items() if v is not None and k not in self.required_parameters}
        
        # Create the model
        self.model = timm.create_model(
            model_name = config['model_name'],
            num_classes = config['num_classes'],
            **kwargs
        )
        
        # OLD VERSION
        # self.model = timm.create_model(
        #     config['model_name'],
        #     pretrained = config['pretrained'],
        #     num_classes = config['num_classes'],
        #     img_size = config['input_size'],
        #     in_chans = config['input_channels'],
        #     embed_dim = config['embed_dim'],
        #     depth = config['depth'],
        #     num_heads = config['num_heads'],
        # )

    def forward(self, x) :
        return self.model(x)

    def classify(self, x, return_prob : bool = False) :
        """
        Classify the input x.
        If return_prob is True, the function will return the probability of each class. Otherwise, it will return the predicted class.

        Parameters
        ----------
        x : torch.tensor
            Input tensor. Shape must be B x C x H x W
        return_prob : bool
            If True, the function will return the probability of each class. Otherwise, it will return the predicted class.

        Returns
        -------
        torch.tensor
            If return_prob is True, the function will return the probability of each class. Otherwise, it will return the predicted class.
            The shape of the output tensor is B x num_classes if return_prob is True. Otherwise, the shape is B.
        """
        x = self.forward(x)

        if return_prob :
            return torch.nn.functional.softmax(x, dim = 1)
        else :
            return torch.argmax(x, dim = 1)
    
    def check_config(self, config : dict) :
        """
        Function to check the model config. By design, only 3 paramters are required to create the model: model_name, pretrained and num_classes. 
        The other parameters are optional and will be passed to the timm.create_model function if they are present in the config dictionary.

        Parameters
        ----------
        config : dict
            The model config dictionary. For more details see `update_model_config_vit.py`.

        Raises
        ------
        ValueError
            If any of the required parameters is missing or has the wrong type.
        """

        # Check if all the required parameters are present in the config and if they have the correct type.
        for parameter in self.required_parameters :
            if parameter not in config :
                raise ValueError(f'Missing required parameter {parameter} in the model config.')
    
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

def get_defaul_config(vit_name : str) :
    """

    """

    if vit_name == 'vit_tiny_patch16_224' :
        pass

