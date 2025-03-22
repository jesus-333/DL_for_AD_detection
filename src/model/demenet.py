"""
Pytorch implementation of DEMNET, A Deep Learning Model for Early Diagnosis of Alzheimer Diseases and Dementia From MR Images.
For more information about the model see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9459692

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch

from . import support_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class demnet(torch.nn.Module) :

    def __init__(self, config : dict) :
        """
        Create the DEMNET model. For more information about the model see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9459692
        
        Parameters
        ----------

        config : dict
            Dictionary with the configuration of the model. The keys of the dictionary are:
            - input_size : int
                Size of the input image. It is assumed that the image is square. This value is used to compute the size of the input of the fully connected layers.
            - input_channels : int
                Number of channels of the input image. It is assumed that the image is grayscale. If not provided, the default value is 3.
            - n_classes : int
                Number of classes of the classification problem.
            - activation : str
                Activation function to use in the model. The activation function must be supported by the support_model.get_activation function.
                Possible values are: 'relu', 'elu', 'selu', 'gelu'.
            - kernel_size_conv_1 : int
                Kernel size of the first convolutional layer. See the scheme of the model in the paper for more information.
            - kernel_size_conv_2 : int
                Kernel size of the second convolutional layer. See the scheme of the model in the paper for more information.
            - batch_norm : bool
                If True, batch normalization is used in the model. If not provided, the default value is True.
            - config_demnet_block_1 : dict
                Configuration of the first demnet block. For more information see the demnet_block init description.
            - config_demnet_block_2 : dict
                Configuration of the second demnet block. For more information see the demnet_block init description.
            - config_demnet_block_3 : dict
                Configuration of the third demnet block. For more information see the demnet_block init description.
            - config_demnet_block_4 : dict
                Configuration of the fourth demnet block. For more information see the demnet_block init description.
            - dropout_rate_1 : float
                Dropout rate of the first dropout layer. If not provided, the default value is 0.5.
            - dropout_rate_2 : float
                Dropout rate of the second dropout layer. If not provided, the default value is 0.5.
            - use_activation_in_classifier : bool
                If True, an activation function is used in the fully connected layers. If not provided, the default value is True.
            - use_as_features_extractor : bool
                If True, the model is used as a feature extractor. If not provided, the default value is False.
                If the model is used as a feature extractor, the output of the model is the output of the convolutional layers, without flattening and classification.
        """
        super(demnet, self).__init__()

        activation = support_model.get_activation(config['activation'])
        
        check_demnet_config(config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Convolutional layers (feature extraction)


        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size = config['kernel_size_conv_1'], padding = 'same'),
            torch.nn.BatchNorm2d(16) if config['batch_norm'] else torch.nn.Identity(),
            activation,
        )

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size = config['kernel_size_conv_2'] , padding = 'same'),
            torch.nn.BatchNorm2d(16) if config['batch_norm'] else torch.nn.Identity(),
            activation,
        )

        self.pool = torch.nn.MaxPool2d(kernel_size = 2)

        self.demnet_block_1 = demnet_block(config['config_demnet_block_1'])
        self.demnet_block_2 = demnet_block(config['config_demnet_block_2'])
        self.demnet_block_3 = demnet_block(config['config_demnet_block_3'])
        self.demnet_block_4 = demnet_block(config['config_demnet_block_4'])

        self.dropout_1 = torch.nn.Dropout2d(config['dropout_rate_1'])
        self.dropout_2 = torch.nn.Dropout2d(config['dropout_rate_2'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Fully connected layers (classification)

        self.flatten = torch.nn.Flatten()

        size_after_conv = self.compute_size_after_conv(config['input_size'], config['input_channels'])
        n_input_neurons = size_after_conv[2] * size_after_conv[3]

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(n_input_neurons, 512),
            activation if config['use_activation_in_classifier'] else torch.nn.Identity(),
            torch.nn.Linear(512, 128),
            activation if config['use_activation_in_classifier'] else torch.nn.Identity(),
            torch.nn.Linear(128, 64),
            activation if config['use_activation_in_classifier'] else torch.nn.Identity(),
            torch.nn.Linear(64, config['n_classes']),
        )

        self.use_as_features_extractor = config['use_as_features_extractor']

    def forward(self, x) :
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.pool(x)

        x = self.demnet_block_1(x)
        x = self.demnet_block_2(x)
        x = self.demnet_block_3(x)
        x = self.dropout_1(x)

        x = self.demnet_block_4(x)
        x = self.dropout_2(x)

        if self.use_as_features_extractor :
            return x
        else :
            x = self.flatten(x)
            x = self.classifier(x)

    def compute_size_after_conv(self, size : int, n_channels : int) :
        use_as_features_extractor_backup = self.use_as_features_extractor
        self.use_as_features_extractor = True

        x = torch.zeros(1, n_channels, size, size)
        x = self(x)

        self.use_as_features_extractor = use_as_features_extractor_backup

        return x.size()

class demnet_block(torch.nn.Module) :

    def __init__(self, config : dict) :
        """
        Create a block of the DEMNET model. For more information about the model see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9459692

        Parameters
        ----------

        config : dict
            Dictionary with the configuration of the model. The keys of the dictionary are:
            - input_channels_block : int
                Number of channels of the input of the block.
            - output_channels_block : int
                Number of channels of the output of the block.
            - kernel_size_conv_1 : int
                Kernel size of the first convolutional layer of the block.
            - kernel_size_conv_2 : int
                Kernel size of the second convolutional layer of the block.
            - batch_norm : bool
                If True, batch normalization is used in the block.
        """
        super(demnet, self).__init__()

        self.activation = support_model.get_activation(config['activation'])

        self.conv_1_block = torch.nn.Conv2d(
            config['input_channels_block'], config['input_channels_block'], 
            kernel_size = config['kernel_size_conv_1'], padding = 'same'
        )

        self.conv_2_block = torch.nn.Conv2d(
                config['input_channels_block'], config['output_channels_block'], 
                kernel_size = config['kernel_size_conv_2'], padding = 'same'
        )

        self.norm = torch.nn.BatchNorm2d(config['output_channels_block']) if config['batch_norm'] else torch.nn.Identity()

        self.pool = torch.nn.MaxPool2d(kernel_size = 2)

    def forward(self, x) :
        x = self.conv_1_block(x)
        x = self.activation(x)
        x = self.conv_2_block(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.pool(x)

        return x

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def check_demnet_config(config : dict) :
    """
    Check if the configuration of the demnet model is correct.

    Parameters
    ----------

    config : dict
        Configuration of the model. The information about the keys of the configuration can be found in the demnet init description and demnet_block init description.
    """
    
    if 'input_size' not in config : raise Exception('Error: input_size not found in config.')
    if config['input_size'] <= 0 : raise Exception(f'Error: input_size must be greater than 0. Current value : {config["input_size"]}')

    if 'input_channels' not in config : 
        print('Warning: input_channels not found in config. Using default value of 3.')
        config['input_channels'] = 3

    if config['input_channels'] <= 0 : raise Exception(f'Error: input_channels must be greater than 0. Current value : {config["input_channels"]}')

    if 'n_classes' not in config : raise Exception('Error: n_classes not found in config.')
    if config['n_classes'] <= 0 : raise Exception(f'Error: n_classes must be greater than 0. Current value : {config["n_classes"]}')

    if 'activation' not in config : raise Exception('Error: activation not found in config. Possible values are: relu, elu, selu, gelu.')

    if 'kernel_size_conv_1' not in config : raise Exception('Error: kernel_size_conv_1 not found in config.')
    if config['kernel_size_conv_1'] <= 0 : raise Exception(f'Error: kernel_size_conv_1 must be greater than 0. Current value : {config["kernel_size_conv_1"]}')

    if 'kernel_size_conv_2' not in config : raise Exception('Error: kernel_size_conv_2 not found in config.')
    if config['kernel_size_conv_2'] <= 0 : raise Exception(f'Error: kernel_size_conv_2 must be greater than 0. Current value : {config["kernel_size_conv_2"]}')

    if 'batch_norm' not in config :
        print('Warning: batch_norm not found in config. Using default value of True.')
        config['batch_norm'] = True

    check_demnet_block_config(config['config_demnet_block_1'])
    check_demnet_block_config(config['config_demnet_block_2'])
    check_demnet_block_config(config['config_demnet_block_3'])
    check_demnet_block_config(config['config_demnet_block_4'])

    if 'dropout_rate_1' not in config :
        print('Warning: dropout_rate_1 not found in config. Using default value of 0.5.')
        config['dropout_rate_1'] = 0.5
    if config['dropout_rate_1'] <= 0 or config['dropout_rate_1'] >= 1 : raise Exception(f'Error: dropout_rate_1 must be between 0 and 1. Current value : {config["dropout_rate_1"]}')

    if 'dropout_rate_2' not in config :
        print('Warning: dropout_rate_2 not found in config. Using default value of 0.5.')
        config['dropout_rate_2'] = 0.5
    if config['dropout_rate_2'] <= 0 or config['dropout_rate_2'] >= 1 : raise Exception(f'Error: dropout_rate_2 must be between 0 and 1. Current value : {config["dropout_rate_2"]}')

    if 'use_activation_in_classifier' not in config :
        print('Warning: use_activation_in_classifier not found in config. Using default value of True.')
        config['use_activation_in_classifier'] = True

    if 'use_as_features_extractor' not in config :
        print('Warning: use_as_features_extractor not found in config. Using default value of False.')
        config['use_as_features_extractor'] = False



def check_demnet_block_config(config : dict) :
    """
    Check if the configuration of the demnet model is correct.

    Parameters
    ----------

    config : dict
        Configuration of the model. The information about the keys of the configuration can be found in the demnet_block init description.
    """
    
    if 'input_channels_block' not in config : raise Exception('Error: input_channels_block not found in config.')
    if config['input_channels_block'] <= 0 : raise Exception(f'Error: input_channels_block must be greater than 0. Current value : {config["input_channels_block"]}')

    if 'output_channels_block' not in config : raise Exception('Error: output_channels_block not found in config.')
    if config['output_channels_block'] <= 0 : raise Exception(f'Error: output_channels_block must be greater than 0. Current value : {config["output_channels_block"]}')

    if 'kernel_size_conv_1' not in config : raise Exception('Error: kernel_size_conv_1 not found in config.')
    if config['kernel_size_conv_1'] <= 0 : raise Exception(f'Error: kernel_size_conv_1 must be greater than 0. Current value : {config["kernel_size_conv_1"]}')

    if 'kernel_size_conv_2' not in config : raise Exception('Error: kernel_size_conv_2 not found in config.')
    if config['kernel_size_conv_2'] <= 0 : raise Exception(f'Error: kernel_size_conv_2 must be greater than 0. Current value : {config["kernel_size_conv_2"]}')

    if 'batch_norm' not in config :
        print('Warning: batch_norm not found in config. Using default value of True.')
        config['batch_norm'] = True








