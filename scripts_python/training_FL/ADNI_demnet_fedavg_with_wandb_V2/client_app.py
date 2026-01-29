"""
Client app for demnet training through FL simulation.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import toml

from flwr.client import ClientApp, Client
from flwr.common import Context

from addl.dataset import support_dataset_ADNI
from addl.federated import client
from addl.model import demnet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def client_fn_demnet(context : Context) -> Client :
    """
    Create a Flower client.
    """

    # Get train and dataset config for the client.
    # On flower slack I found a reply that say that the node_config should be used for the deployment while the run_config for the simulation
    dataset_config  = toml.load(context.run_config["path_dataset_config"])
    model_config    = toml.load(context.run_config["path_model_config"])
    training_config = toml.load(context.run_config["path_training_config"])
    if 'print_var' not in training_config : training_config['print_var'] = False
    
    # Set the seed
    # NOT USED NOW. Since I launch all the training from the sh script, before the flwr command I called the update_trainig.py script that already set the seed in the toml file
    # training_config['seed'] = context.run_config["seed"] if 'seed' in context.run_config else np.random.randint(0, 2**32)
    # import pprint
    # print("training config for cleint", context.node_config["partition-id"])
    # pprint.pprint(training_config)

    # Get client ID (it is a number from 0 to n_client - 1) and load indices for the client
    # Note that at the moment (21/01/26) it is not possible to set node-config during simulation (at the best of my knowledge).
    client_id = context.node_config["partition-id"]
    idx_client = np.load(dataset_config['path_idx_folder'] + f'train_idx_client_{client_id}.npy')

    MRI_train_dataset, MRI_validation_dataset, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = dataset_config['percentage_train'], idx_to_use = idx_client, seed = training_config['seed'])
    
    if training_config['print_var'] :
        print("#######################################")
        print("CLIENT")
        print("dataset_config['merge_AD_class'] ", dataset_config['merge_AD_class'])
        print(f"N. training samples    : {len(MRI_train_dataset)}")
        print(f"N. validations samples : {len(MRI_validation_dataset)}")
        print("#######################################")

    # (OPTIONAL) Move dataset to device
    if dataset_config['load_data_in_memory'] :
        MRI_train_dataset.move_data_and_labels_to_device(training_config['device'])
        MRI_validation_dataset.move_data_and_labels_to_device(training_config['device'])
        # MRI_test_dataset.move_data_and_labels_to_device(training_config['device'])
    
    # Add client id to dictionary (so ti can be saved inside the class)
    training_config['client_id'] = client_id

    # Load model
    model_config['input_channels'] = MRI_train_dataset.data.shape[1]
    # model_config['num_classes'] = len(np.unique(labels_int)) # Move to the sh file
    model = demnet.demnet(model_config)

    return client.flower_client_v1(model, MRI_train_dataset, MRI_validation_dataset, training_config).to_client()

app = ClientApp(client_fn_demnet)
