"""
Implementation for the Volume-Attention ViT Architecture

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
import math

from .vit import timm_vit

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# vav implementation

class vav(torch.nn.Module) :
    """
    Volume-Attention ViT architecture.

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
        In this case, the model will produce three separate embeddings for the whole volume, one for each direction, which will then fuseded together (e.g. by concatenation) and passed through a classification head to produce the final output.
        If False, a single transformer encoder will be used to process the concatenation of the three sequences of embeddings produced by the ViTs for all three directions.
        In this case, the model will produce a single embedding for the whole volume, which will then be passed through a classification head to produce the final output.
    """

    def __init__(self, config : dict) :
        super().__init__()
        
        # Crate the ViT(s) for each direction to extract the slice embeddings.
        if config['use_same_vit_for_all_directions'] :
            self.vit_axial = vit_encoder(config['vit_config'])
            self.vit_sagittal = self.vit_axial
            self.vit_coronal = self.vit_axial
        else :
            self.vit_axial    = vit_encoder(config['vit_config'])
            self.vit_sagittal = vit_encoder(config['vit_config'])
            self.vit_coronal  = vit_encoder(config['vit_config'])
        
        # Output dimension of the ViT encoder, which will be the input dimension for the transformer encoder(s).
        vit_encoder_output = self.vit_axial.vit.embed_dim
        config['slice_transformer_config']['d_input'] = vit_encoder_output
        
        # Store the analyze_directions_independently flag to use it in the forward pass.
        self.analyze_directions_independently = config['analyze_directions_independently']

        # Create the transformer encoder(s) to process the sequence of embeddings produced by the ViT for each direction and produce a single embedding for the whole volume.
        # Note that for a torch.nn.TransformerEncoder, the output dimension is the same as the input dimension.
        if config['analyze_directions_independently'] :
            self.encoder_axial    = slice_transformer_encoder(config['slice_transformer_config'])
            self.encoder_sagittal = slice_transformer_encoder(config['slice_transformer_config'])
            self.encoder_coronal  = slice_transformer_encoder(config['slice_transformer_config'])

            # Since we will concatenate the three embeddings produced by the three transformer encoders, the output dimension of the concatenated embedding will be 3 times the output dimension of each transformer encoder.
            output_dim = 3 * config['slice_transformer_config']['d_model']
        else :
            self.encoder = slice_transformer_encoder(config['slice_transformer_config'])
            output_dim = config['slice_transformer_config']['d_model']

        # Classification head to produce the final output.
        self.classifier = torch.nn.Linear(output_dim, config['num_classes'])

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.tensor
            Input tensor of shape B x X x Y x Z, where B is the batch size, X, Y and Z are the dimensions of the input volume.
            The input tensor follow the NifTi convention, where the first dimension (X) corresponds to the sagittal direction, the second dimension (Y) corresponds to the coronal direction and the third dimension (Z) corresponds to the axial direction.

        Returns
        -------
        torch.tensor
            Output tensor of shape B x num_classes, where B is the batch size and num_classes is the number of output classes.
        """

        # Compute the embeddings for each direction using the corresponding ViT. The input x is expected to be a dictionary with keys 'axial', 'sagittal' and 'coronal', each containing a tensor of shape B x C x H x W, where B is the batch size, C is the number of channels, H and W are the height and width of the slices in that direction.
        embeddings_axial    = self.vit_axial(x)
        embeddings_sagittal = self.vit_sagittal(x)
        embeddings_coronal  = self.vit_coronal(x)

        if self.analyze_directions_independently :
            # Process the embeddings for each direction with the corresponding transformer encoder to produce a single embedding for the whole volume for that direction. 
            # The resulting tensors will have shape B x D, where D is the embedding dimension of the transformer encoder.
            volume_embedding_axial    = self.encoder_axial(embeddings_axial)
            volume_embedding_sagittal = self.encoder_sagittal(embeddings_sagittal)
            volume_embedding_coronal  = self.encoder_coronal(embeddings_coronal)

            # Concatenate the embeddings for all three directions along the feature dimension.
            # The resulting tensor will have shape B x (3*D), where D is the embedding dimension of the transformer encoder.
            volume_embedding = torch.cat([volume_embedding_axial, volume_embedding_sagittal, volume_embedding_coronal], dim = 1)
        else :
            # Concatenate the embeddings for all three directions along the sequence dimension. The resulting tensor will have shape B x (S_axial + S_sagittal + S_coronal) x D, where S is the number of slices in each direction and D is the embedding dimension of the ViT.
            embeddings = torch.cat([embeddings_axial, embeddings_sagittal, embeddings_coronal], dim = 1)

            # Process the concatenated embeddings with the transformer encoder to produce a single embedding for the whole volume. The resulting tensor will have shape B x D, where D is the embedding dimension of the transformer encoder.
            volume_embedding = self.encoder(embeddings)

        # Classification
        output = self.classifier(volume_embedding)

        return output

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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ViT encoder implementation

class vit_encoder(torch.nn.Module) :
    """
    ViT encoder used to extract the slice embeddings for each direction. It used the ViT implementation in the vit.py module, which is based on the timm library.
    
    This module expect the input to be a tensor of shape B x S x H x W, where B is the batch size, S is the number of slices, H and W are the height and width of the slices in that direction.
    The output of this module is a tensor of shape B x S x D, where S is the number of slices in that direction and D is the embedding dimension of the ViT.
    """

    def __init__(self, config : dict) :
        super().__init__()

        # We don't need a classification head for the ViT encoder, since we only want the embeddings.
        # Setting the num_classes to 0 will make the timm create a ViT model without a classification head, i,e. the output of the ViT will be the embeddings.
        config['num_classes'] = 0
        
        # Create the ViT
        self.vit = timm_vit(config)

    def forward(self, x) :
        """
        Forward pass of the ViT encoder.

        Parameters
        ----------
        x : torch.tensor
            Input tensor of shape B x S x H x W, where B is the batch size
        """
        # Create a tensor to store the embeddings for each slice. The shape of this tensor is B x S x D, where B is the batch size, S is the number of slices and D is the embedding dimension of the ViT.
        # embeddings = torch.zeros(x.shape[0], x.shape[1], self.vit.model.embed_dim, device = x.device)
        batch_size, num_slices, height, width = x.shape
        
        # Reshape tensor x to have shape (B*S) x 1 x H x W. This way we can pass all the slices through the ViT at once exploiting the batch dimension, instead of iterating over the slices and passing them one by one through the ViT, which would be more inefficient.
        x = x.view(-1, 1, height, width)

        # Compute the embeddings for all slices at once
        embeddings = self.vit(x)

        # Reshape the embeddings tensor to have shape B x S x D, where B is the batch size, S is the number of slices and D is the embedding dimension of the ViT.
        embeddings = embeddings.view(batch_size, num_slices, -1)

        return embeddings

    def check_reshape_trick(self, x : torch.Tensor, use_eval_mode : bool = True, use_double_precision : bool = False) :
        """
        Function to check if the reshape trick used in the forward pass to compute the embeddings for all slices at once is equivalent to iterating over the slices and passing them one by one through the ViT.
        
        Note that the first time I use this function I hadn't add the double precision, and I got a difference around 1e-6 between the two methods.
        This SHOULD BE expected, since the two methods perform the computations in a slightly different way, and this SHOULD have some impact on the numerical precision of the output.
        The double precision option was added exactly to reduce this difference. In all my tests with double precision I get a difference of 0.

        Parameters
        ----------
        x : torch.tensor
            Input tensor of shape B x S x H x W, where B is the batch size, S is the number of slices, H and W are the height and width of the slices in that direction.
        use_eval_mode : bool
            If True, the model will be set to evaluation mode before computing the embeddings, which will disable dropout and other training-specific layers that could introduce randomness in the output and make the comparison between the two methods not valid. Default is True.
        use_double_precision : bool
            If True, the model and the input tensor will be converted to double precision before computing the embeddings, which will reduce the numerical differences between the two methods that could be caused by the order of operations and the way the computations are performed in the two methods.
            If used the model will be set back to float precision after the check, and the input tensor will be converted back to float precision as well. Default is False.
        """
        
        # Set the model to evaluation mode to disable dropout and other training-specific layers, which could introduce randomness in the output and make the comparison between the two methods not valid.
        if use_eval_mode :
            self.eval()
        
        # Use double precision to reduce the numerical differences between the two methods, which could be caused by the order of operations and the way the computations are performed in the two methods. 
        # This is especially important when comparing the outputs of two different methods, as small numerical differences can accumulate and lead to larger differences in the final output.
        if use_double_precision :
            self.double()
            x = x.double()

        emb_1 = self.forward(x)

        emb_2 = torch.zeros_like(emb_1)
        for i in range(x.shape[1]) :
            emb_2[:, i, :] = self.vit(x[:, i, :, :].unsqueeze(1))

        print(f"Max absolute difference between the two methods  : {torch.max(torch.abs(emb_1 - emb_2))}")
        print(f"Mean absolute difference between the two methods : {torch.mean(torch.abs(emb_1 - emb_2))}")
        
        # Reset the model to training mode and float precision after the check
        if use_double_precision :
            self.float()
            x = x.float()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Slice encoder implementation

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
        super().__init__()

        # Layers for input projection and positional encoding
        self.input_proj = torch.nn.Linear(config['d_input'], config['d_model'])
        self.pos_enc = positional_encoding(config['d_model'], max_len = config['max_len'], dropout = config['dropout'])
        
        # Transformer encoder layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model         = config['d_model'],
            nhead           = config['nhead'],
            dim_feedforward = config['dim_feedforward'],
            dropout         = config['dropout'],
            activation      = config['activation'],
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

