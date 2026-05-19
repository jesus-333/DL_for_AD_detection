"""
Implementation for the Attention Volume Vit Architecture

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
import math

from .vit import timm_vit

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# avv implementation

class avv(torch.nn.Module) :
    """
    Attention Volume ViT architecture.

    This architecture is designed to work with 3D MRI volumes. For each direction (axial, sagittal, coronal), a ViT is applied to the slices of the volume in that direction, producing a sequence of embeddings. 
    These sequences are then processed by a transformer to produce a single embedding for the whole volume, which is then passed through a classification head to produce the final output.

    The model is configured through a config dictionary, which must contain the following fields :
    - use_same_vit_for_all_directions : bool
        If True, the same ViT will be used for all three directions. If False, a separate ViT will be used for each direction.
    - vit_config : dict
        The configuration dictionary for the ViT model. This dictionary will be passed to the timm_vit function to create the ViT models for each direction.
        Note that each ViT model will have the same configuration, regardless of whether use_same_vit_for_all_directions is True or False.
    - analyze_directions_independently : bool
        If True, a separate transformer encoder will be used for each direction to process the sequence of embeddings produced by the ViT for that direction.
        In this case, the model will prodice three separate embeddings for the whole volume, one for each direction, which will then fuseded together (e.g. by concatenation) and passed through a classification head to produce the final output.
        If False, a single transformer encoder will be used to process the concatenation of the three sequences of embeddings produced by the ViTs for all three directions.
        In this case, the model will produce a single embedding for the whole volume, which will then be passed through a classification head to produce the final output.
    """

    def __init__(self, config : dict) :
        super().__init__()
        
        # Crate the ViT(s) for each direction to extract the slice embeddings.
        if config['use_same_vit_for_all_directions'] :
            self.vit_axial = timm_vit(config['vit_config'])
            self.vit_sagittal = self.vit_axial
            self.vit_coronal = self.vit_axial
        else :
            self.vit_axial    = timm_vit(config['vit_config'])
            self.vit_sagittal = timm_vit(config['vit_config'])
            self.vit_coronal  = timm_vit(config['vit_config'])
        
        # TODO : Modify config to ensure correct input dimensions for the transformer encoder(s) based on the output dimension of the ViT(s) and the number of slices in each direction.
        if config['analyze_directions_independently'] :
            self.encoder_axial    = slice_transformer_classifier(config['slice_transformer_config'])
            self.encoder_sagittal = slice_transformer_classifier(config['slice_transformer_config'])
            self.encoder_coronal  = slice_transformer_classifier(config['slice_transformer_config'])
        else :
            self.encoder = slice_transformer_classifier(config['slice_transformer_config'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submodules implementation

class positional_encoding(torch.nn.Module):
    def __init__(self, d_model : int, max_len : int = 512, dropout : float = 0.1):
        super().__init__()
    
        self.dropout = torch.nn.Dropout(dropout)
            
        # Create the positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Tensor of shape (max_len, 1) with values from 0 to max_len-1. Used to compute the positional encoding for each position in the sequence.
        position = torch.arange(max_len).unsqueeze(1).float()
        
        # Compute frequncies for the positional encoding. The frequencies are computed as 1 / (10000^(2i/d_model)) for i in [0, d_model/2-1].
        # This means that the frequencies decrease exponentially with the dimension of the model, allowing the model to capture both short-term and long-term dependencies in the sequence.
        # Note that we compute d_model/2 frequencies, since we will use the sine function for the even dimensions and the cosine function for the odd dimensions of the positional encoding matrix.
        # The log is used to avoid numerical instability when computing the frequencies for large values of d_model.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Fill the even dimensions of the positional encoding matrix with the sine of the position multiplied by the corresponding frequency.
        # Since position has shape (max_len, 1) and div_term has shape (d_model/2), the multiplication will broadcast the frequencies across all positions, resulting in a matrix of shape (max_len, d_model/2).
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Same as above, but for the odd dimensions of the positional encoding matrix, which are filled with the cosine of the position multiplied by the corresponding frequency.
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding matrix.
        # register_buffer means this tensor is part of the module state (it moves to GPU with .cuda(), gets saved/loaded), but it's not a learnable parameter, i.e. the optimizer won't touch it.
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class slice_transformer_encoder(torch.nn.Module):
    def __init__(self, config : dict) :
        """
        Transformer used to process the sequence of ViT embeddings per slice and produce a single embedding for the whole volume.
        """

        # Config fields
        # d_input     : int,
        # num_classes : int,
        # d_model     : int = 256,
        # nhead       : int = 8,
        # num_layers  : int = 4,
        # dim_feedforward : int = 512,
        # dropout     : float = 0.1,

        super().__init__()

        # Layers for input projection and positional encoding
        self.input_proj = torch.nn.Linear(config['d_input'], config['d_model'])
        self.pos_enc = positional_encoding(config['d_model'], dropout = config['dropout'])
        
        # Transformer encoder layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model         = config['d_model'],
            nhead           = config['n_head'],
            dim_feedforward = config['dim_feedforward'],
            dropout         = config['dropout'],
            batch_first     = True,
        )

        # Transformer model (i.e. n layers of the encoder)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers = config['num_layers'])

    def forward(self, x, mask = None):
        """
        """

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask = mask)

        # pool over the sequence — mean of non-padded tokens
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)

        return x

class slice_transformer_classifier(slice_transformer_encoder) :
    def __init__(self, config : dict) :
        """
        Extension of the slice_transformer_encoder with a classification head. The output of the transformer is passed through a linear layer to produce the final classification output.
        """
         
        super().__init__()

        self.encoder = slice_transformer_encoder(config)
        self.head = torch.nn.Linear(config['d_model'], config['num_classes'])

    def forward(self, x, mask = None):
        x = self.encoder(x, mask)
        x = self.head(x)
        return x

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

