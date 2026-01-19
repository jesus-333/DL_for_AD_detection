"""
Note that addl is the name of the package built through hatch from the file in the src folder.

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
from addl.model import demnet
from addl.training import test_functions

# (TMP) Added for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def gen_evaluate_fn(model, idx_data_server, all_config : dict) :
    """
    Generate the function for centralized evaluation.
    """
    
    # Get dataset and training config
    dataset_config  = all_config['dataset_config']
    training_config = all_config['training_config']

    # Transform data and label in the dataset
    test_dataset, _, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = 1, idx_to_use = idx_data_server, seed = training_config['seed'])

    def evaluate(server_round, parameters_ndarrays, config):
        """
        Evaluate global model on centralized test set.
        """
        # Set the model weights
        support_federated_generic.set_weights(model, parameters_ndarrays)
        
        # Transform data and label in the dataset
        test_dataset, _, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = 1, idx_to_use = idx_data_server, seed = training_config['seed'])

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

def server_fn(context : Context) :
    """
    Create and return the ServerAppComponents required by Flower
    """

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
    idx_data_per_client = []
    
    # Save (separately) data for central evaluation
    if all_config['server_config']['centralized_evaluation'] :
        idx_data_server = np.load(server_config["path_data_idx_server"])
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
    model = demnet.demnet(model_config)

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
        evaluate_fn        = gen_evaluate_fn(model, idx_data_server, all_config) if all_config['server_config']['centralized_evaluation'] else None,
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
