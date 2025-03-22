"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch

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
    n_parameters = sum(p.numel() for p in  layer.parameters() if p.requires_grad)
    return n_parameters
