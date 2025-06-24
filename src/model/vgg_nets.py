"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

from . import download_published_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class VGG(torch.nn.Module):
    
    def __init__(self, model, num_classes : int, input_channels : int = 3) :
        """
        Create a VGG network. The model must be a VGG network downloaded through the download_vgg_nets function in the download_published_model module.

        Parameters
        ----------

        model : torch.nn.Module
            VGG network model. As defined in https://pytorch.org/vision/main/models/vgg.html
        num_classes : int
            Number of classes of the dataset
        input_channels : int
            Number of input channels of the model. Default is 3 (i.e. RGB image). If set to 1, the first layer of the model will be modified to accept single channel input (i.e. grayscale image). Could be set to number higher than 3 if you use a different type of input (e.g. MRI scans where each channel is a different slice of the scan). Default is 3.
        """

        super(VGG, self).__init__()

        # Modify the last layer to have the correct number of classes
        model.classifier[-1] = torch.nn.Linear(4096, num_classes)

        # (OPTIONAL) Modify the first layer to accept single channel input
        if input_channels != 3 :
            model.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        print("Nota that the input must be in the rescaled between 0 and 1")

    def forward(self, x) :
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

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

    def set_training_model(self, training_mode : int = 0) :
        """
        Set the training modality according to the input parameter training_mode
        - 0 : All the layer will be trained
        - 1 : Only the last layer (i.e. the classifier[6] layer) will be trained/finetuned
        - 2 : All the classifier layers will be trained/finetuned
        - 3 : Only the first and last layer will be trained/finetuned.

        Parameters
        ----------
        training_mode : int
            Type of finetuning. Possible values are 0, 1, 2, or 3.
        """

        if training_mode == 0 :
            for param in self.parameters() : param.requires_grad = True

        elif training_mode == 1 :
            self.freeze_model()

            for param in self.classifier[6].parameters() :
                param.requires_grad = True

        elif training_mode == 2 :
            self.freeze_model()

            for param in self.classifier.parameters() :
                param.requires_grad = True

        elif training_mode == 3 :
            self.freeze_model()

            for param in self.features[0].parameters() :
                param.requires_grad = True

            for param in self.classifier[6].parameters() :
                param.requires_grad = True

    def freeze_model(self) :
        """
        Freeze all the layers of the model. This means that the parameters of the model will not be updated during training.
        """

        for param in self.features.parameters() : param.requires_grad = False
        for param in self.avgpool.parameters() : param.requires_grad = False
        for param in self.classifier.parameters() : param.requires_grad = False

    def check_freeze_layer(self) :
        """
        Check, for each layer, if the layer is freezed or not (i.e. if the layer requires gradient or not)
        """
        for name, param in self.named_parameters():
            print(name, "\t", param.requires_grad)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_vgg(config : dict, return_preprocess_functions : bool = True) :
    """
    Create and return a VGG network and the preprocess functions.

    Parameters
    ----------

    config : dict
        Configuration of the model. The configuration must contain the following keys:
        - version : int
            Version of the VGG network. Possible values are 11, 13, 16 or 19
        - batch_normalization : bool
            If True, the model will have batch normalization
        - pretrained : bool
            If True, the model will be pretrained
        - input_channels : int
            Number of input channels of the model. Input channels are the number of channels of the input images, e.g. 3 for RGB images, 1 for grayscale images, or higher than 3 for other types of input (e.g. MRI scans where each channel is a different slice of the scan).
        - num_classes : int
            Number of classes of the dataset
    return_preprocess_functions : bool
        If True, the function will return also the preprocessing functions to use with the model. Otherwise, it will return only the model.

    Returns
    -------
    model : VGG
        The VGG network model, implemented through the VGG class defined in this module.
    preprocess_functions : torchvision.transforms.Compose
        The preprocessing functions to use with the model. This variable will be returned only if return_preprocess_functions is True. Otherwise, only the model will be returned.
        Note that the preprocessing functions are torchvision.transforms.Compose() object, that Resize, CenterCrop and Normalize the input images.
        The preprocessing functions are defined in the download_published_model module. They expect to receive input images in the range [0, 1] (i.e. rescaled images) with 3 depth channels (i.e. RGB images).
    """
    
    # Check config
    check_model_config(config)

    # Get the model
    model, preprocess_functions = download_published_model.download_vgg_nets(config['version'], config['batch_normalization'], config['pretrained'])

    # Create the model
    model = VGG(model, config['num_classes'], config['input_channels'])
    
    if return_preprocess_functions :
        return model, preprocess_functions
    else :
        return model

def check_model_config(config : dict) :
    """
    Check if the configuration of the model is correct.

    Parameters
    ----------

    config : dict
        Configuration of the model. The information about the keys of the configuration can be found in the get_vgg function description.
    """

    if 'version' not in config :
        raise ValueError('The version key is missing in the configuration')

    if config['version'] not in [11, 13, 16, 19] :
        raise ValueError(f"The version key must be 11, 13, 16 or 19. The current value is {config['version']}")

    if 'batch_normalization' not in config :
        raise ValueError('The batch_normalization key is missing in the configuration')

    if 'pretrained' not in config :
        raise ValueError('The pretrained key is missing in the configuration')

    if 'input_channels' not in config :
        print('The input_channels key is missing in the configuration. It will be set to 3')
        config['input_channels'] = 3
    
    if config['input_channels'] <= 0 :
        raise ValueError(f'The input_channels key must be greater than 0. Current value is {config["input_channels"]}')

    if 'num_classes' not in config :
        raise ValueError('The num_classes key is missing in the configuration')

    if config['num_classes'] <= 0 :
        raise ValueError(f'The num_classes key must be greater than 0. Current value is {config["num_classes"]}')


