"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pandas as pd
import toml
import torch

from flwr.server import ServerConfig, ServerAppComponents, ServerApp
from flwr.common import Context, ndarrays_to_parameters

from addl.dataset import support_dataset_ADNI
from addl.federated import server, support_federated_generic
from addl.model import download_published_model, vgg_nets
from addl.training import test_functions

# (TMP) Added for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def gen_evaluate_fn(model, idx_data_server, all_config : dict, preprocess_functions) :
    """
    Generate the function for centralized evaluation.
    """
    
    # Get dataset and training config
    dataset_config  = all_config['dataset_config']
    training_config = all_config['training_config']
    
    # Transform data and label in the dataset
    # test_dataset, _, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = 1, idx_to_use = idx_data_server, seed = training_config['seed'], preprocess_functions = preprocess_functions)

    def evaluate(server_round, parameters_ndarrays, config):
        """
        Evaluate global model on centralized test set.
        """
        # Set the model weights
        support_federated_generic.set_weights(model, parameters_ndarrays)
        
        # Transform data and label in the dataset
        test_dataset, _, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = 1, idx_to_use = idx_data_server, seed = training_config['seed'], preprocess_functions = preprocess_functions)

        # Evaluate the model on test data
        test_loss, test_metrics_dict = test_functions.test(training_config, model, test_dataset, label = 'server')

        return test_loss, test_metrics_dict

    return evaluate

def gen_on_fit_config_fn(all_config : dict) :
    """
    Generate the function to update the config when the fit method is called by the client
    """

    def on_fit_config(server_round : int) :
        """
        Construct `config` that clients receive when running `fit()`
        """

        lr = all_config['server_config']['starting_lr_per_round'][server_round]

        return {"lr" : lr}

    return on_fit_config

def prepare_data_for_FL_training(all_config : dict) :
    """
    Read the data and get the path to all of them, split uniformly between clients and save them in npy files.
    For the ADNI dataset, if I use single tensor file to store the data, I split the indices.
    I.e. I create an array with value from 0 to n - 1 (with n = total number of samples). After that, the array is split, so for each client I have a list of indices. When I load the data for a client I load only the data corresponding to the indices of that client.
    
    Note that the indices are saved in npy files that are later read by the client.
    The returned indices have only the purpose to allow their backup in the dictionaries uploaded in wandb.
    """

    dataset_config = all_config['dataset_config']
    # dataset_name = dataset_config['dataset_name']
    dataset_tensor_file_name = dataset_config['name_tensor_file']
    path_to_data = dataset_config['path_data']

    dataset_info = pd.read_csv(f'{path_to_data}dataset_info.csv')
    labels_int = dataset_info['labels_int'].to_numpy()

    # Set the number of clients.
    # If I use the centralized_evaluation, I add an extra client for the server. In this way the data will be split in n + 1 parts.
    # n parts will be used for the clients and the last one will be used for the server.
    n_client = all_config['server_config']['n_client'] if not all_config['server_config']['centralized_evaluation'] else all_config['server_config']['n_client'] + 1

    data = torch.load(f'{path_to_data}{dataset_tensor_file_name}', mmap = True)
    idx = np.arange(len(data))

    # Split the data uniformly between clients
    # In this case, I only interested in the split of the indices, not the data/labels itself.The data/labels will be loaded later in the client.
    # Note that even if I not keep the labels here, I still pass them to the function, so I can use the keep_labels_proportion parameter.
    idx_per_client, _ = support_federated_generic.split_data_for_clients_uniformly(idx, n_client = n_client,
                                                                                   seed = all_config['training_config']['seed'],
                                                                                   labels = labels_int, keep_labels_proportion = all_config['server_config']['keep_labels_proportion']
                                                                                   )
    
    # Create the folder to save the data
    os.makedirs(all_config['dataset_config']['path_data'], exist_ok = True)

    # Save data and labels in npy files
    for i in range(all_config['server_config']['n_client']) :
        # Path to save indices
        dataset_path = all_config['dataset_config']['path_data'] + f'{i}_idx.npy'

        # Save indices
        np.save(dataset_path, idx_per_client[i])
    
    # If centralized_evaluation is True, save the indices for the server
    if all_config['server_config']['centralized_evaluation'] :
        dataset_path = all_config['dataset_config']['path_data'] + 'server_idx.npy'
        np.save(dataset_path, idx_per_client[-1])

    return idx_per_client

def server_fn(context : Context) :
    """
    Create and return the ServerAppComponents required by Flower
    """

    # In this you want to check the working directory
    # cwd = os.getcwd()
    # print(cwd)
    # print(context)
    # Apparently, even if it print as working directory the one where you launch the flwr run command, it search for other files to the folder where pyproject.toml is located

    # Get all the config dictionaries
    dataset_config  = toml.load(context.run_config["path_dataset_config"])
    model_config    = toml.load(context.run_config["path_model_config"])
    server_config   = toml.load(context.run_config["path_server_config"])
    training_config = toml.load(context.run_config["path_training_config"])
    if 'print_var' not in training_config : training_config['print_var'] = False

    # Get seed 
    if training_config['seed'] == -1 : training_config['seed'] = np.random.randint(0, 2**32 - 1)
    
    # Create single config dictionary
    all_config = dict(
        dataset_config   = dataset_config,
        model_config     = model_config,
        server_config    = server_config,
        training_config  = training_config
    )
    
    # Server/strategy parameters
    num_rounds    = server_config["num_rounds"]
    fraction_fit  = server_config["fraction_fit"]
    fraction_eval = server_config["fraction_evaluate"]
    if len(server_config['wandb_config']['metrics_to_log_from_clients']) == 0 : server_config['wandb_config']['metrics_to_log_from_clients'] = None

    # Prepare dataset for FL training and central evaluation
    idx_data_per_client = prepare_data_for_FL_training(all_config)
    
    # Save (separately) data for central evaluation
    if all_config['server_config']['centralized_evaluation'] :
        idx_data_server = idx_data_per_client[-1]
    else :
        idx_data_server = labels_server = None

    # Backup of the indices for the various clients (and server)
    all_config['backup_dataset'] = dict(
        idx_data_per_client = idx_data_per_client,
        idx_data_server     = idx_data_server,
    )

    # Get the number of labels
    dataset_info = pd.read_csv(f'{dataset_config['path_data']}dataset_info.csv')
    labels_int, labels_str = dataset_info['labels_int'].to_numpy(), dataset_info['labels_str'].to_numpy()
    labels_int = support_dataset_ADNI.merge_AD_class_function(labels_int, labels_str, dataset_config['merge_AD_class'])
    num_classes  = len(np.unique(labels_int))

    # Update model config with the number of classes
    model_config['num_classes'] = num_classes

    # Create model
    model = vgg_nets.get_vgg(model_config, return_preprocess_functions = False)

    # Get Customize Preprocess Functions
    if dataset_config['use_normalization'] :
        # Get mean and std for normalization
        if training_config['use_vgg_normalization_values'] :
            preprocess_functions = download_published_model.get_preprocess_functions('vgg')

            # Save them in the config (In this way I save them also on wandb)
            dataset_config['mean_dataset'] = torch.tensor([0.485, 0.456, 0.406])
            dataset_config['std_dataset']  = torch.tensor([0.229, 0.224, 0.225])
        else :
            mean_dataset = torch.load(f'{dataset_config['path_data']}dataset_mean.pt')
            std_dataset  = torch.load(f'{dataset_config['path_data']}dataset_std.pt')
            preprocess_functions = download_published_model.get_preprocess_functions('vgg', mean = mean_dataset, std = std_dataset)

            # Save them in the config (In this way I save them also on wandb)
            dataset_config['mean_dataset'] = mean_dataset
            dataset_config['std_dataset']  = std_dataset
    else :
        preprocess_functions = None

    # Initialize model parameters
    ndarrays = support_federated_generic.get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = server.fed_avg_with_wandb_tracking(
        model              = model,
        all_config         = all_config,
        fraction_fit       = fraction_fit,
        fraction_evaluate  = fraction_eval,
        initial_parameters = parameters,
        evaluate_fn        = gen_evaluate_fn(model, idx_data_server, all_config, preprocess_functions) if all_config['server_config']['centralized_evaluation'] else None,
        fit_metrics_aggregation_fn      = support_federated_generic.weighted_average,
        evaluate_metrics_aggregation_fn = support_federated_generic.weighted_average,
        on_fit_config_fn = gen_on_fit_config_fn(all_config) if all_config['server_config']['use_on_fit_config_function'] else None
    )
    config = ServerConfig(num_rounds = num_rounds)

    if training_config['print_var'] :
        print("#######################################")
        print("SERVER APP")
        print(model_config)
        print("#######################################")

    return ServerAppComponents(strategy = strategy, config = config)

# Create ServerApp
app = ServerApp(server_fn = server_fn)
