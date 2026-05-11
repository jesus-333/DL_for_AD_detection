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
            embed_dim = config['embed_dim'],
            depth = config['depth'],
            num_heads = config['num_heads'],
        )

    def forward(self, x) :
        return self.model(x)

