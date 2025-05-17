"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import torch

from flwr.client import NumPyClient

from . import support_federated_generic
from ..training import train_functions, test_functions

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class flower_client_v1(NumPyClient):
    def __init__(self, model, train_dataset, validation_dataset, training_config):
        # Save model, dataset_config and training_config
        self.model = model
        self.training_config = training_config
        if 'print_var' not in self.training_config : self.training_config['print_var'] = False

        # Save dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # Check if client id is saved inside training_config
        # Note that usually the client_id is not speciefied in the toml file. It is added during the client creation in through the client_fn functions
        # E.g. see the function client_fn_demnet in scripts/training_FL/fedavg_with_wandb/client_app.py
        self.client_id = training_config["client_id"] if 'client_id' in training_config else -1
        if self.client_id == -1 : print('WARNING: client id not found in training_config. Set to -1')

        # Update path to save weights with client id
        self.training_config['path_to_save_model'] += f'/client_{self.client_id}'

        # Print information
        if self.training_config['print_var'] :
            print(f"Client n. {self.client_id}")
            print(f"N. training samples = {len(self.train_dataset)}")
            print(f"N. validation_dataset samples = {len(self.validation_dataset)}")

    def get_weights(self, config):
        return support_federated_generic.get_weights(self.model)

    def fit(self, parameters, config):
        # Set the parameters (received from the server)
        support_federated_generic.set_weights(self.model, parameters)

        # Update training config
        for key in config :
            if key in self.training_config :
                self.training_config[key] = config[key]

        # Train the model
        if self.training_config['print_var'] : print(f"START TRAINING Client {self.client_id}")
        self.model, training_metrics = train_functions.train(self.training_config, self.model, self.train_dataset, self.validation_dataset)
        if self.training_config['print_var'] : print(f"END TRAINING Client {self.client_id}")

        # Convert metrics (see notes in the function description)
        converted_training_metrics = self.convert_training_metrics_for_upload(training_metrics)
        
        # Add extra information to the metrics
        converted_training_metrics['client_id'] = self.client_id
        converted_training_metrics['epochs']    = self.training_config['epochs']

        # (OPTIONAL) Load the weights that obtain the lower validation error (I.e. early stop)
        if self.training_config['use_weights_with_lower_validation_error'] :
            path_weights_early_stop = os.path.join(self.training_config['path_to_save_model'], 'model_BEST.pth') 
            self.model.load_state_dict(torch.load(path_weights_early_stop, map_location = self.training_config['device']))

        return support_federated_generic.get_weights(self.model), len(self.train_dataset), converted_training_metrics

    # def evaluate(self, parameters, config) :
    #     """
    #     NOT USED FOR NOW
    #     """
    #
    #     print("Evaluate function not implemented for the client. Returning dummy values.")
    #
    #     # support_federated.set_weights(self.model, parameters)
    #     # loss, accuracy = test_functions.test()
    #     # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    #
    #     return 1.0, len(self.validation_dataset), {"accuracy": 1.0}

    def convert_training_metrics_for_upload(self, training_metrics) :
        """
        For now flower want the returned metrics (i.e. the third elemented returned be the fit method) to be scalars.
        To keep track of how the metrics change during training I saved them as list. This is a workaround to convert them in a scalar.
        In the server implementation there is a similar method that merge the single scalar again in a new list.
        """

        converted_training_metrics = dict()
    
        # Save metrics over time
        for metric in training_metrics :
            # Avoid conversion of confusion matrix
            if 'confusion_matrix' in metric : continue

            # Iterate over length epoch
            for i in range(self.training_config['epochs']) :
                converted_training_metrics[f'{metric}:{i}'] = training_metrics[metric][i]

            # Duplicate metrics at last epoch (so it easy to recover the final metrics values)
            converted_training_metrics[f'{metric}:END'] = training_metrics[metric][i]

        return converted_training_metrics

