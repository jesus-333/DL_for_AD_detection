"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import toml

import torch

from flwr.client import Client, NumPyClient
from flwr.common import Context

from . import support_federated
from ..training import train_functions, test_functions
from ..model import demenet
from ..dataset import dataset, support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class flower_client_v1(NumPyClient):
    def __init__(self, model, train_dataset, validation_dataset, train_config, dataset_config):
        # Save model, dataset_config and train_config
        self.model = model
        self.train_config = train_config
        self.dataset_config = dataset_config

        # Save dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def get_weights(self, config):
        return support_federated.get_weights(self.model)

    def fit(self, parameters, config):
        # Set the parameters (received from the server) 
        support_federated.set_weights(self.model, parameters)

        # Train the model
        self.model, training_metrics = train_functions.train(self.train_config, self.model, self.train_dataset, self.validation_dataset)

        # Convert metrics (see notes in the function description)
        converted_training_metrics = self.convert_training_metrics_for_upload(training_metrics)

        return support_federated.get_weights(self.model), len(self.trainloader), converted_training_metrics

    def evaluate(self, parameters, config) :
        """
        NOT USED FOR NOW
        """

        support_federated.set_weights(self.model, parameters)
        loss, accuracy = test_functions.test()
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def convert_training_metrics_for_upload(self, training_metrics) :
        """
        For now flower want the returned metrics (i.e. the third elemented returned be the fit method) to be scalars.
        To keep track of how the metrics change during training I saved them as list. This is a workaround to convert them in a scalar.
        In the server implementation there is a similar method that merge the single scalar again in a new list.
        """

        converted_training_metrics = []

        for metric in training_metrics :
            for i in range(self.train_config['epochs']) :
                converted_training_metrics[f'{metric}_{i}'] = training_metrics[metric][i]

        return converted_training_metrics

def client_fn_demnet(context : Context) -> Client :
    """
    Create a Flower client representing a single organization.
    """

    # TODO modify path to save the model in train config
    
    # Get model config. Note that the model must be the same for all clients
    model_config   = context.node_config['model_config']

    # Get train and dataset config for the client. Unlike the models each client could have its own custom config regarding data and client.
    # (IMHO, it is better to uniform also these settings but, just in case, I'd like the idea to have the possibility to have different settings for each client)
    train_config   = toml.load(context.node_config['train_config_path'])
    dataset_config = toml.load(context.node_config['dataset_config_path'])

    # Load model
    model = demenet.demnet(model_config)

    # Load data and labels
    file_path_list_client, label_list_int_client = support_federated.load_data_and_labels(context.node_config["partition-id"], dataset_config)
    
    # Get idx to split the  data in train/validation/test
    idx_list = support_dataset.get_idx_to_split_data_V3(label_list_int_client, dataset_config['percentage_split_train_val'], train_config['seed'])
    idx_train, idx_validation = idx_list
    
    # Split data in train/validation/test
    train_file_path_list,      label_train_list_int      = file_path_list_client[idx_train],      label_list_int_client[idx_train]
    validation_file_path_list, label_validation_list_int = file_path_list_client[idx_validation], label_list_int_client[idx_validation]

    # Create dataset
    # Note that the dataloaders are created directly inside the training function.
    load_data_in_memory = dataset_config['load_data_in_memory']
    train_dataset      = dataset.MRI_2D_dataset(train_file_path_list, label_train_list_int,           load_data_in_memory = load_data_in_memory, preprocess_functions = dataset_config['preprocess_functions'], grey_scale_image = dataset_config['grey_scale_image'])
    validation_dataset = dataset.MRI_2D_dataset(validation_file_path_list, label_validation_list_int, load_data_in_memory = load_data_in_memory, preprocess_functions = dataset_config['preprocess_functions'], grey_scale_image = dataset_config['grey_scale_image'])

    return flower_client_v1(model, train_dataset, validation_dataset, train_config, dataset_config).to_client()
