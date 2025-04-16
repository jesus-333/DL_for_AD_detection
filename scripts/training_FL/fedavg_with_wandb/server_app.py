"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports 

import toml
import os
import numpy as np

from flwr.server import ServerConfig, ServerAppComponents, ServerApp
from flwr.common import Context, ndarrays_to_parameters

from src.model import demnet
from src.dataset import dataset, support_dataset, support_dataset_kaggle
from src.federated import server, support_federated_generic
from src.training import test_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def gen_evaluate_fn(model, data, labels, all_config : dict) :
    """
    Generate the function for centralized evaluation.
    """
    
    # Get the various config dict
    dataset_config  = all_config['dataset_config']
    model_config    = all_config['model_config']
    training_config = all_config['training_config']

    def evaluate(server_round, parameters_ndarrays, config):
        """
        Evaluate global model on centralized test set.
        """
        # Set the model weights
        support_federated_generic.set_weights(model, parameters_ndarrays)
        
        # Trasform data and label in the dataset 
        test_dataset = support_dataset_kaggle.get_dataset_with_preprocess_function_from_data(data, labels, model_config, dataset_config)

        # Evaluate the model on test data
        test_loss, test_metrics_dict = test_functions.test(training_config, model, test_dataset)

        return test_loss, test_metrics_dict  

    return evaluate

def prepare_data_for_FL_training(all_config : dict) :
    """
    Read the data and get the path to all of them, split uniformly between clients and save them in npy files.
    The path where to store the npy file is specified inside dataset_config['path_data']
    """
    path_files_Moderate_Demented    = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/ModerateDemented'
    path_files_Mild_Demented        = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/MildDemented'
    path_files_Very_Mild_Demented   = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/VeryMildDemented'
    path_files_Non_Demented         = './data/Kaggle_Alzheimer_MRI_4_classes_dataset/NonDemented'

    file_path_list, label_list_int, label_list_str = support_dataset_kaggle.get_data(path_files_Moderate_Demented, path_files_Mild_Demented, path_files_Very_Mild_Demented, path_files_Non_Demented, 
                                                                                        all_config['dataset_config']['merge_AD_class'], print_var = True
                                                                                        )

    # Set the number of clients. 
    # If I use the centralized_evaluation, I add an extra client for the server. In this way the the data will be split in n + 1 parts.
    # n parts will be used for the clients and the last one will be used for the server.
    n_client = all_config['server_config']['n_client'] if not all_config['server_config']['centralized_evaluation'] else all_config['server_config']['n_client'] + 1

    # Split the data uniformly between clients
    data_per_client, labels_per_client = support_federated_generic.split_data_for_clients_uniformly(file_path_list, n_client = all_config['server_config']['n_client'], 
                                                                                                    seed = all_config['training_config']['seed'], labels = label_list_int, 
                                                                                                    keep_labels_proportion = all_config['server_config']['keep_labels_proportion'] 
                                                                                                    )
    
    # Create the folder to save the data
    os.makedirs(all_config['dataset_config']['path_data'], exist_ok = True)

    # Save data and labels in npy files
    for i in range(all_config['server_config']['n_client']) :
        # Path to save data and labels
        dataset_path = all_config['dataset_config']['path_data'] + f'{i}_data.npy'
        labels_path  = all_config['dataset_config']['path_data'] + f'{i}_labels.npy'

        # Save data and labels
        np.save(dataset_path, data_per_client[i])
        np.save(labels_path, labels_per_client[i])
    
    # If centralized_evaluation is True, save the data and labels for the server
    if all_config['server_config']['centralized_evaluation'] :
        dataset_path = all_config['dataset_config']['path_data'] + 'server_data.npy'
        labels_path  = all_config['dataset_config']['path_data'] + 'server_labels.npy'
        np.save(dataset_path, data_per_client[-1])
        np.save(labels_path, labels_per_client[-1])

    return data_per_client, labels_per_client

def server_fn(context : Context):
    # Get all the config dictionaries
    dataset_config  = toml.load(context.run_config["path_dataset_config"])
    model_config    = toml.load(context.run_config["path_model_config"])
    server_config   = toml.load(context.run_config["path_server_config"])
    training_config = toml.load(context.run_config["path_training_config"])

    # Get seed 
    training_config['seed'] = context.run_config["seed"] if 'seed' in context.run_config else None
    training_config['seed'] = np.random.randint(0, 10e8)
    
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
    if len(server_config['metrics_to_log_from_clients']) == 0 : server_config['metrics_to_log_from_clients'] = None 

    # Prepare dataset for FL training and central evaluation
    data_per_client, labels_per_client = prepare_data_for_FL_training(all_config)
    
    # Save (separately) data for central evaluation
    if all_config['server_config']['centralized_evaluation'] :
        data_server = data_per_client[-1]
        labels_server = labels_per_client[-1]
    else :
        data_server = labels_server = None

    # Backup the dataset
    all_config['backup_dataset'] = dict(
        data_per_client = data_per_client,
        labels_per_client = labels_per_client,
        data_server = data_server,
        labels_server = labels_server
    )

    # Create model
    model = demnet.demnet(model_config)

    # Initialize model parameters
    ndarrays = support_federated_generic.get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = server.fed_avg_with_wandb_tracking(
        model = model,
        all_config = all_config,
        fraction_fit = fraction_fit,
        fraction_evaluate = fraction_eval,
        initial_parameters = parameters,
        # on_fit_config_fn = on_fit_config, # TODO in future iteration
        evaluate_fn = gen_evaluate_fn(model, data_server, labels_server, all_config) if all_config['server_config']['centralized_evaluation'] else None,
        fit_metrics_aggregation_fn = support_federated_generic.weighted_average,
        evaluate_metrics_aggregation_fn = support_federated_generic.weighted_average,
    )
    config = ServerConfig(num_rounds = num_rounds)

    return ServerAppComponents(strategy = strategy, config = config)


# Create ServerApp
app = ServerApp(server_fn = server_fn)
