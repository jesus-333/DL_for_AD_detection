"""

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import torchvision
# import timm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class swin_transformer_classifier(torch.nn.Module) :
    """
    Swin transformer classifier for image classification tasks. It is composed by a swin transformer encoder and a fully connected layer for classification.
    The version and size of the swin transformer can be specified in the constructor. The forward method takes an input image and returns the output of the fully connected layer.

    Parameters
    ----------
    num_classes : int
        Number of classes for the classification task.
    version : int, optional
        Version of the swin transformer model. It can be either 1 or 2. Default is 2.
    size : str, optional
        Size of the swin transformer model. It can be either 'tiny', 'small' or 'base'. Default is 'tiny'.

    Attributes
    ----------
    version : int
        Version of the swin transformer model. It can be either 1 or 2.
    size : str
        Size of the swin transformer model. It can be either 'tiny', 'small' or 'base'.
    encoder : torch.nn.Module
        Swin transformer encoder for feature extraction created according to the version and size specified in the constructor. It is implmemented using the torchvision library.
    classifier : torch.nn.Linear
        Fully connected layer for classification.
    transforms : torchvision.transforms
        Transformations to be applied to the input image before passing it through the encoder. It is obtained automatically when the swin transformer model is created.
    apply_transforms_to_input : bool
        If True, the transforms of the swin transformer model are applied to the input image before passing it through the encoder. Default is False.

    TODO add possibility to use model not pretrained
    """

    def __init__(self, num_classes : int, version : int = 2, size : str = 'tiny') :
        super(swin_transformer_classifier, self).__init__()
        
        # Set the version and size of the swin transformer model. It can be either 1 or 2 for the version and 'tiny', 'small' or 'base' for the size.
        self.set_version_and_size(version, size, update_model = False)
        
        # Construct the swin transformer model
        self.set_swin_model()

        # Construct the fully connected layer for classification.
        if num_classes <= 0 : raise ValueError(f"num_classes must be a positive integer. Current value is {num_classes}")
        classifier = torch.nn.Linear(self.encoder.head.in_features, num_classes)
        self.encoder.head = classifier
        
        # Others attributes
        self.apply_transforms_to_input = False

    def forward(self, x) :
        """
        Forward method for the swin transformer classifier. It takes an input image and returns the output of the fully connected layer.
        If apply_transforms_to_input is True, the transforms of the swin transformer model are applied to the input image before passing it through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes)
        """
        if self.apply_transforms_to_input :
            x = self.transforms(x)

        x = self.encoder(x)

        x = self.classifier(x)

        return x

    def set_training_mode(self, training_mode : int = 0) :
        """
        Set the training modality according to the input parameter training_mode
        - 0 : All the layer will be trained
        - 1 : Only the last layer (i.e. the head) will be trained/finetuned

        Parameters
        ----------
        training_mode : int
            Type of finetuning. Possible values are 0, 1.
        """

        if training_mode == 0 :
            for param in self.parameters() :
                param.requires_grad = True
        elif training_mode == 1 :
            for param in self.parameters() :
                param.requires_grad = False
            for param in self.encoder.head.parameters() :
                param.requires_grad = True
        else :
            raise ValueError(f"training_mode must be either 0 or 1. Current value is {training_mode}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def set_swin_model(self) :
        """
        Set the swin transformer model according to the version and size specified in the constructor.
        """

        if self.version == 1 :
            self.encoder, self.transforms = get_swin_model_v1(self.size, return_transforms = True)
        elif self.version == 2 :
            self.encoder, self.transforms = get_swin_model_v2(self.size, return_transforms = True)

    def set_version_and_size(self, version : int, size : str, update_model : bool = True) :
        """
        Set the version and size of the swin transformer model. It can be either 1 or 2 for the version and 'tiny', 'small' or 'base' for the size.
        """

        self.check_version_and_size(version, size)
        self.version = version
        self.size = size
        if update_model : self.set_swin_model()

    def set_version(self, version : int, update_model : bool = True) :
        """
        Set the version of the swin transformer model. It can be either 1 or 2.
        """

        self.check_version_and_size(version, self.size)
        self.version = version
        if update_model : self.set_swin_model()

    def set_size(self, size : str, update_model : bool = True) :
        """
        Set the size of the swin transformer model. It can be either 'tiny', 'small' or 'base'.
        """

        self.check_version_and_size(self.version, size)
        self.size = size
        if update_model : self.set_swin_model()

    def check_version_and_size(self, version : int, size : str) :
        if version not in [1, 2] :
            raise ValueError(f"version must be either 1 or 2. Current value is {version}")
        if size not in ['tiny', 'small', 'base'] :
            raise ValueError(f"size must be either 'tiny', 'small' or 'base'. Current value is {size}")

def get_swin_model_v1(size : str = 'tiny', return_transforms : bool = True) :
    if size == 'tiny' :
        weight_obj = torchvision.models.Swin_T_Weights.DEFAULT
        swin_model = torchvision.models.swin_t(weights = weight_obj)
    elif size == 'small' :
        weight_obj = torchvision.models.Swin_S_Weights.DEFAULT
        swin_model = torchvision.models.swin_s(weights = weight_obj)
    elif size == 'base' :
        weight_obj = torchvision.models.Swin_B_Weights.DEFAULT
        swin_model = torchvision.models.swin_b(weights = weight_obj)

    if return_transforms :
        transforms = weight_obj.transforms()
        return swin_model, transforms
    else :
        return swin_model

def get_swin_model_v2(size : str = 'tiny', return_transforms : bool = True) :
    if size == 'tiny' :
        weight_obj = torchvision.models.Swin_V2_T_Weights.DEFAULT
        swin_model = torchvision.models.swin_v2_t(weights = weight_obj)
    elif size == 'small' :
        weight_obj = torchvision.models.Swin_V2_S_Weights.DEFAULT
        swin_model = torchvision.models.swin_v2_s(weights = weight_obj)
    elif size == 'base' :
        weight_obj = torchvision.models.Swin_V2_B_Weights.DEFAULT
        swin_model = torchvision.models.swin_v2_b(weights = weight_obj)

    if return_transforms :
        transforms = weight_obj.transforms()
        return swin_model, transforms
    else :
        return swin_model
