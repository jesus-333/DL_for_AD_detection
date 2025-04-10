"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports 

import toml

from flwr.client import ClientApp, Client
from flwr.common import Context

from src.model import demnet
from src.dataset import support_dataset, dataset
from src.federated import client, support_federated_generic

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def client_fn_demnet(context : Context) -> Client :
    """
    Create a Flower client. 
    """
    
    # Get model config. Note that the model must be the same for all clients
    model_config = context.run_config['model_config']

    # Get train and dataset config for the client.
    training_config   = toml.load(context.run_config['path_training_config'])
    dataset_config = toml.load(context.run_config['path_dataset_config'])
    # training_config   = toml.load(context.node_config['training_config_path'])
    # dataset_config = toml.load(context.node_config['dataset_config_path'])
    # On flower slack I found a reply that say that the node_config should be used for the deployment. The run_config for the simulation

    
    training_config['seed'] = context.run_config["seed"] if 'seed' in context.run_config["seed"] else None

    # Load model
    model = demnet.demnet(model_config)

    # Load data and labels
    # The data and the labels must be saved in npy files in the path specified in dataset_config['path_data'].
    # For each client there must be two files: one for the data and one for the labels.
    # The files must be named as follows : x_data.npy and x_labels.npy, where x is the client_id.
    client_id = context.node_config["partition-id"]
    dataset_path = dataset_config['path_data'] + f'{client_id}_data.npy'
    labels_path = dataset_config['path_data'] + f'{client_id}_labels.npy'
    file_path_list_client, label_list_int_client = support_federated_generic.load_data_and_labels(dataset_path, labels_path, dataset_config)
    
    # Get idx to split the  data in train/validation/test
    idx_list = support_dataset.get_idx_to_split_data_V3(label_list_int_client, [dataset_config['percentage_split_train_val']], training_config['seed'])
    idx_train, idx_validation = idx_list
    
    # Split data in train/validation/test
    train_file_path_list,      label_train_list_int      = file_path_list_client[idx_train],      label_list_int_client[idx_train]
    validation_file_path_list, label_validation_list_int = file_path_list_client[idx_validation], label_list_int_client[idx_validation]

    # Create dataset (Note that the dataloaders are created directly inside the training function)
    load_data_in_memory = dataset_config['load_data_in_memory']
    train_dataset      = dataset.MRI_2D_dataset(train_file_path_list, label_train_list_int, load_data_in_memory = load_data_in_memory, 
                                                preprocess_functions = dataset_config['preprocess_functions'], grey_scale_image = dataset_config['grey_scale_image']
                                                )

    validation_dataset = dataset.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, 
                                                preprocess_functions = dataset_config['preprocess_functions'], grey_scale_image = dataset_config['grey_scale_image']
                                                )

    return client.flower_client_v1(model, train_dataset, validation_dataset, training_config).to_client()

app = ClientApp(client_fn_demnet)
