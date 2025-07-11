"""
Pytorch implementation of DEMNET, A Deep Learning Model for Early Diagnosis of Alzheimer Diseases and Dementia From MR Images.
For more information about the model see https://ieeexplore.ieee.org/abstract/document/9459692

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
import matplotlib.pyplot as plt
import os

try :
    import pytorch_grad_cam
    gradcam_available = True
except : 
    print("Warning: pytorch-grad-cam not found. The XAI functions will not work. You can install the package with 'pip install grad-cam'")
    print('See here for more details : https://github.com/jacobgil/pytorch-grad-cam')
    gradcam_available = False
try :
    import wandb
    wandb_available = True
except :
    print("Warning: wandb not found. The functions to download the pretrained model from wandb will not work. You can install the package with 'pip install wandb'")
    wandb_available = False

from . import support_model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class demnet(torch.nn.Module) :
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
        - num_classes : int
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

    Attributes
    ----------
    conv_1 : torch.nn.Module
        First convolutional layer of the model.
    conv_2 : torch.nn.Module
        Second convolutional layer of the model.
    pool : torch.nn.Module
        Max pooling layer of the model.
    demnet_block_1 : torch.nn.Module
        First demnet block of the model. For more information see the demnet_block init description and the original paper.
    demnet_block_2 : torch.nn.Module
        Second demnet block of the model. For more information see the demnet_block init description and the original paper.
    demnet_block_3 : torch.nn.Module
        Third demnet block of the model. For more information see the demnet_block init description and the original paper.
    demnet_block_4 : torch.nn.Module
        Fourth demnet block of the model. For more information see the demnet_block init description and the original paper.
    dropout_1 : torch.nn.Module
        First dropout layer of the model.
    dropout_2 : torch.nn.Module
        Second dropout layer of the model.
    flatten : torch.nn.Module
        Flatten layer of the model. Used to flatten the output of the convolutional layers before passing it to the fully connected layers.
    classifier : torch.nn.Module
        Fully connected layers of the model. Used to classify the output of the convolutional layers.
    use_as_features_extractor : bool
        If True, the model is used as a feature extractor. If not provided, the default value is False.
        If the model is used as a feature extractor, the output of the model is the output of the convolutional layers, without flattening and classification.
    """

    def __init__(self, config : dict) :
        super(demnet, self).__init__()

        check_demnet_config(config)
        activation = support_model.get_activation(config['activation'])
        self.use_as_features_extractor = False

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Convolutional layers (feature extraction)

        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(config['input_channels'], 16, kernel_size = config['kernel_size_conv_1'], padding = 'same'),
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
        n_input_neurons = size_after_conv[1] * size_after_conv[2] * size_after_conv[3]

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(n_input_neurons, 512),
            activation if config['use_activation_in_classifier'] else torch.nn.Identity(),
            torch.nn.Linear(512, 128),
            activation if config['use_activation_in_classifier'] else torch.nn.Identity(),
            torch.nn.Linear(128, 64),
            activation if config['use_activation_in_classifier'] else torch.nn.Identity(),
            torch.nn.Linear(64, config['num_classes']),
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

    def compute_size_after_conv(self, size : int, n_channels : int) :
        use_as_features_extractor_backup = self.use_as_features_extractor
        self.use_as_features_extractor = True

        x = torch.zeros(1, n_channels, size, size)
        x = self(x)

        self.use_as_features_extractor = use_as_features_extractor_backup

        return x.size()

    def print_input_size_after_each_conv_layer(self, size : int, n_channels : int) :
        print(f"Input size : {n_channels}x{size}x{size}")

        x = torch.zeros(1, n_channels, size, size)
        x = self.conv_1(x)
        print(f"Size after conv_1 :\t\t{x.size()}")
        x = self.conv_2(x)
        print(f"Size after conv_2 :\t\t{x.size()}")
        x = self.pool(x)
        print(f"Size after pool :\t\t{x.size()}")
        x = self.demnet_block_1(x)
        print(f"Size after demnet_block_1 :\t{x.size()}")
        x = self.demnet_block_2(x)
        print(f"Size after demnet_block_2 :\t{x.size()}")
        x = self.demnet_block_3(x)
        print(f"Size after demnet_block_3 :\t{x.size()}")
        x = self.dropout_1(x)
        print(f"Size after dropout_1 :\t\t{x.size()}")
        x = self.demnet_block_4(x)
        print(f"Size after demnet_block_4 :\t{x.size()}")
        x = self.dropout_2(x)
        print(f"Size after dropout_2 :\t\t{x.size()}")

        if not self.use_as_features_extractor :
            x = self.flatten(x)
            print(f"Size after flatten :\t\t{x.size()}")

    def print_trainable_parameters(self) :
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters : {trainable_parameters}")

    def compute_gradcam_for_input(self, x, target_layer : str, target_class : int) :
        """
        Get the Grad-CAM of the input x.

        Parameters
        ----------
        x : torch.tensor
            Input tensor. Shape must be B x C x H x W
        target_layer : str 
            Layer to use to compute the Grad-CAM. The possible values are : conv_1, conv_2, demnet_block_1, demnet_block_2, demnet_block_3, demnet_block_4
        target_class : int
            Class to use to compute the Grad-CAM.

        Returns
        -------
        torch.tensor
            Grad-CAM of the input x. The shape of the output tensor is B x H x W.
        """
        if not gradcam_available :
            print('Error: pytorch-grad-cam not found. The XAI functions will not work.')
            print('No output will be returned.')
            print('Please install pytorch-grad-cam with the command : pip install pytorch-grad-cam and restart the script.')
        else :
            # gradcam = pytorch_grad_cam.GradCAM(model = self, target_layers = [self._modules[target_layer]])
            # self.gradcam = pytorch_grad_cam.GradCAM(model = self, target_layers = [self.demnet_block_4.pool])

            target_class = [pytorch_grad_cam.utils.model_targets.ClassifierOutputTarget(target_class)]
            grayscale_cam = self.gradcam(x, target_class)

            return grayscale_cam

    def visualize_gradcame_for_input(self, x : torch.tensor, target_layer : str, target_class : int, figsize : tuple = (10, 5), cmap : str = 'gray', alpha : float = 0.3) :
        """
        Visualize the Grad-CAM of the input x.
        Use the function compute_gradcam_for_input to compute the Grad-CAM.

        Parameters
        ----------
        x : torch.tensor
            Input tensor. Shape must be 1 x C x H x W
        target_layer : str
            Layer to use to compute the Grad-CAM. The possible values are : conv_1, conv_2, demnet_block_1, demnet_block_2, demnet_block_3, demnet_block_4
        target_class : int
            Class to use to compute the Grad-CAM.
        figsize : tuple
            Size of the figure. Default value is (10, 5).
        cmap : str
            Color map of the input image. Default value is 'gray'.
        alpha : float
            Alpha value of the Grad-CAM. Default value is 0.3.
        """

        # Compute the Grad-CAM
        grayscale_cam = self.compute_gradcam_for_input(x, target_layer, target_class)

        # Create the figure
        fig, ax = plt.subplots(1, 1, figsize = figsize)
    
        # Plot the input image and the Grad-CAM
        ax.imshow(x.squeeze().cpu().detach().numpy(), cmap = cmap)
        ax.imshow(grayscale_cam.squeeze(), cmap = 'jet', alpha = alpha)
        
        # Show the figure
        fig.tight_layout()
        plt.show()

        return fig, ax

    def set_training_mode(self, training_mode : int = 0) :
        """
        This function is used to freeze (some) of the model parameters. This is useful when you want to finetune the model.
        To frozen the parameters, the requires_grad attribute of the parameters is set to False.
        The layers that are frozen are decided by the training_mode parameter. The possible values are:
        - 0 : No layaer will be frozen, i.e. all the layers will be trained.
        - 1 : Only the last layer of the classifier will be trained. The rest of the model will be frozen.
        - 2 : All the classifier layers will be trained. The rest of the model will be frozen.

        Parameters
        ----------
        training_mode : int
            Type of finetuning. Possible values are 0, 1 or 2.
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

    def freeze_model(self) :
        for param in self.parameters() : param.requires_grad = False

    def check_freeze_layer(self) :
        """
        Check, for each layer, if the layer is freezed or not (i.e. if the layer requires gradient or not)
        """
        self.print_debug(0)

    def check_average_parameters_value_per_layer(self) :
        """
        This method is used to check the average value of the parameters of each layer.
        It was implmented beacause I noticed that during training with some datasets, I noticed a gradient near zero from the beginning of the training.
        So basically this methods is used for debugging purposes.
        """
        self.print_debug(1)

    def check_average_grad_value_per_layer(self) :
        """
        This method is used to check the average value of the gradients of each layer.
        It was implmented beacause I noticed that during training with some datasets, I noticed a gradient near zero from the beginning of the training.
        So basically this methods is used for debugging purposes.
        """
        self.print_debug(2)

    def print_debug(self, print_mode : int = 0) :
        str_to_print = ""
        for name, param in self.named_parameters():
            str_to_print += name + "\n"

            if print_mode == 0 : # Print all info
                str_to_print += f"\tRequire grad  : {param.requires_grad}\n"
                str_to_print += f"\tAverage param : {float(param.mean())}\n"
                str_to_print += f"\tAverage grad  : {float(param.grad.mean())}\n"
            elif print_mode == 1 : # Print only require grad
                str_to_print += f"\tRequire grad  : {param.requires_grad}\n"
            elif print_mode == 2 : # Print average param
                str_to_print += f"\tAverage param : {float(param.mean())}\n"
            elif print_mode == 3 : # Print average grad
                str_to_print += f"\tAverage grad  : {float(param.grad.mean())}\n"

        print(str_to_print)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
        super(demnet_block, self).__init__()

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

def get_model_pretrained_with_kaggle_dataset(version : int, load_early_stop_weights : bool = True, epoch_to_load : int = -1) -> (torch.nn.Module, dict) :
    """
    Using wandb artifact, download the pretrained model of the DEMNET trained with the Kaggle Alzheimer 4 classes dataset.
    To use this function you need to have wandb installed. You can install the package with 'pip install wandb'.
    The runs information can be visualized at this link : https://wandb.ai/jesus_333/demnet_training/
    The version input specify the version of the artifact to download. Notable versions are:
    - v5 : Trained to classify all the original 4 classes : NonDemented, VeryMildDemented, MildDemented, ModerateDemented
    - v7 : The VeryMildDemented and MildDemented classes were merged into a single class before training. Therefore, the model is trained to classify only three classes : NonDemented, MildDemented, ModerateDemented
    - v8 : All dementia class were merged into a single class before training. Therefore, the model is trained to classify only two classes : NonDemented, Demented
    
    Parameters
    ----------
    version : int
        Version of the artifact to download.
    load_early_stop_weights : bool
        If True, the weights of the model at the early stop epoch are loaded. Default is True. If False, you must specify the epoch to load with the epoch_to_load parameter.
    epoch_to_load : int
        If load_early_stop_weights is False, specify the epoch to load. If load_early_stop_weights is True, this parameter is ignored. Otherwise, must be a number greater than 0.

    Returns :
    model : torch.nn.Module
        Pretrained model of the DEMNET trained with the Kaggle Alzheimer 4 classes dataset.
    metadata : dict
        Metadata of the artifact. Contains all the information about the configuration of the model, dataset and training process. The keys of the dictionary are:
        - model_config : dict
            Configuration of the model. The information about the keys of the configuration can be found in the demnet init description and demnet_block init description.
        - dataset_config : dict
            Configuration of the dataset used to train the model.
        - training_config : dict
            Configuration of the training process.
    """

    if load_early_stop_weights == False and epoch_to_load <= 0 :
        raise Exception('Error: epoch_to_load must be greater than 0 if load_early_stop_weights is False. Current value : {epoch_to_load}')

    # Download the artifact
    run = wandb.init()
    artifact = run.use_artifact(f'jesus_333/demnet_training/demnet_training_AD_kaggle_trained:v{version}', type='model')
    artifact_dir = artifact.download()

    # Get the path to model weights
    # file_list = os.listdir(artifact_dir)

    # Create demnet
    model = demnet(artifact.metadata['model_config'])

    # Load weights
    if load_early_stop_weights :
        model.load_state_dict(torch.load(os.path.join(artifact_dir, 'model_BEST.pth'), map_location = torch.device('cpu')))
    else :
        model.load_state_dict(torch.load(os.path.join(artifact_dir, f'model_{epoch_to_load}.pth'), map_location = torch.device('cpu')))

    return model, artifact.metadata

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

    if 'num_classes' not in config : raise Exception('Error: num_classes not found in config.')
    if config['num_classes'] <= 0 : raise Exception(f'Error: num_classes must be greater than 0. Current value : {config["num_classes"]}')

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
