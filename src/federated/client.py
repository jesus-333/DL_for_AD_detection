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

from . import support_federated_generic
from ..training import train_functions, test_functions
from ..model import demnet
from ..dataset import dataset, support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class flower_client_v1(NumPyClient):
    def __init__(self, model, train_dataset, validation_dataset, train_config):
        # Save model, dataset_config and train_config
        self.model = model
        self.train_config = train_config

        # Save dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

    def get_weights(self, config):
        return support_federated_generic.get_weights(self.model)

    def fit(self, parameters, config):
        # Set the parameters (received from the server) 
        support_federated_generic.set_weights(self.model, parameters)

        # Train the model
        self.model, training_metrics = train_functions.train(self.train_config, self.model, self.train_dataset, self.validation_dataset)

        # Convert metrics (see notes in the function description)
        converted_training_metrics = self.convert_training_metrics_for_upload(training_metrics)

        return support_federated_generic.get_weights(self.model), len(self.trainloader), converted_training_metrics

    def evaluate(self, parameters, config) :
        """
        NOT USED FOR NOW
        """

        # support_federated.set_weights(self.model, parameters)
        # loss, accuracy = test_functions.test()
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def convert_training_metrics_for_upload(self, training_metrics) :
        """
        For now flower want the returned metrics (i.e. the third elemented returned be the fit method) to be scalars.
        To keep track of how the metrics change during training I saved them as list. This is a workaround to convert them in a scalar.
        In the server implementation there is a similar method that merge the single scalar again in a new list.
        """

        converted_training_metrics = []
    
        # Save metrics over time
        for metric in training_metrics :
            for i in range(self.train_config['epochs']) :
                converted_training_metrics[f'{metric}:{i}'] = training_metrics[metric][i]

            # Duplicate metrics at last epoch (so it easy to recover the final metrics values)
            converted_training_metrics[f'{metric}:END'] = training_metrics[metric][i]

        return converted_training_metrics

