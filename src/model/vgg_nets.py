"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch

from . import download_published_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class VGG(torch.nn.Module):
    
    def __init__(self, model, num_classes : int, use_single_channel_input : bool = False) :
        """
        Create a VGG network. The model must be a VGG network downloaded through the download_vgg_nets function in the download_published_model module.

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
            - use_singla_channel_input
                If True, the firts layer of the model will be modified to accept single channel input
            - num_classes : int
                Number of classes of the dataset

        """

        super(VGG, self).__init__()

        # Modify the last layer to have the correct number of classes
        model.classifier[6] = torch.nn.Linear(4096, num_classes)

        # (OPTIONAL) Modify the first layer to accept single channel input
        if use_single_channel_input :
            model.features[0] = torch.nn.Conv2d(1, 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1))

        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.use_single_channel_input = use_single_channel_input

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

    def set_model_for_finetuning(self, finetuning_type : int = 0) :
        """
        Set the model for finetuning. The model can be set in three different ways, through the finetuning_type parameter:
        - 0 : The model will be set to finetune all the layers
        - 1 : The model will be set to finetune only the last layer
        - 2 : The model will be set to finetune only the first and last layer. This option is valid only if use_single_channel_input is True (i.e. the first layer has been modified))

        Parameters
        ----------
        finetuning_type : int
            Type of finetuning. Possible values are 0, 1 or 2.
        """

        if finetuning_type == 0 :
            for param in self.parameters() : param.requires_grad = True

        elif finetuning_type == 1 :
            self.freeze_model()

            for param in self.classifier[6].parameters() :
                param.requires_grad = True

        elif finetuning_type == 2 :
            if not self.use_single_channel_input :
                raise ValueError('The finetuning_type 2 is valid only if use_single_channel_input is True')

            self.freeze_model()

            for param in self.features[0].parameters() :
                param.requires_grad = True

            for param in self.classifier[6].parameters() :
                param.requires_grad = True

    def freeze_model(self) :
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

def get_vgg(config : dict) :
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
        - use_singla_channel_input
            If True, the firts layer of the model will be modified to accept single channel input
        - num_classes : int
            Number of classes of the dataset
    """
    
    # Check config
    check_model_config(config)

    # Get the model
    model, preprocess_functions = download_published_model.download_vgg_nets(config['version'], config['batch_normalization'], config['pretrained'])

    # Create the model
    model = VGG(model, config['num_classes'], config['use_single_channel_input'])

    return model, preprocess_functions 


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

    if 'use_single_channel_input' not in config :
        print('The use_single_channel_input key is missing in the configuration. It will be set to False')
        config['use_single_channel_input'] = False

    if 'num_classes' not in config :
        raise ValueError('The num_classes key is missing in the configuration')

    if config['num_classes'] <= 0 :
        raise ValueError(f'The num_classes key must be greater than 0. Current value is {config["num_classes"]}')


