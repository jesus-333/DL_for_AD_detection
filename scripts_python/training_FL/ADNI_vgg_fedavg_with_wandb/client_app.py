"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import toml
import torch

from flwr.client import ClientApp, Client
from flwr.common import Context

from addl.dataset import support_dataset_ADNI
from addl.federated import client
from addl.model import download_published_model, support_model, vgg_nets

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def client_fn_vgg(context : Context) -> Client :
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

    # Get client ID (it is a number from 0 to n_client - 1) and load indices for the client
    client_id = context.node_config["partition-id"]

    # Idx path
    path_to_load_idx = os.path.join(dataset_config['path_data'], str(training_config['seed']) + '/')

    # Load idx
    idx_client = np.load(path_to_load_idx + f'{client_id}_idx.npy')

    # Get the preprocess functions config for the VGG model
    preprocess_functions_config = download_published_model.get_preprocess_functions_config_for_specific_model('vgg')

    # Get mean and std for normalization
    if dataset_config['use_normalization'] :
        # Check if the user want to use the precomputed mean and std values from the ImageNet dataset or its own values
        if 'use_vgg_normalization_values' in training_config : # Check if the parameter use_vgg_normalization_values is inside the training_config
            if not training_config['use_vgg_normalization_values'] :
                # If the parameter is False, it is assumed that the user want to use its own mean and std values
                load_user_value_for_mean_and_std = True
            else :
                # If the parameter is True the precomputed values from the ImageNet Dataset are already inside the preprocess_functions_config
                load_user_value_for_mean_and_std = False
        else :
            # If use_vgg_normalization_values is not present in the training_config but use_normalization is True, it is assumed that the user want to use its own mean and std values
            load_user_value_for_mean_and_std = True

        if load_user_value_for_mean_and_std:
            # The mean and std values for normalization must be saved inside two tensor files, in the same location of the data
            preprocess_functions_config['mean'] = torch.load(f'{dataset_config['path_data']}dataset_mean.pt')
            preprocess_functions_config['std'] = torch.load(f'{dataset_config['path_data']}dataset_std.pt')
    else :
        # If use_normalization is False then mean and std will be removed by the config
        del preprocess_functions_config['mean']
        del preprocess_functions_config['std']

    # Get the preprocess functions for VGG model
    preprocess_functions = support_model.get_preprocess_functions(preprocess_functions_config)

    MRI_train_dataset, MRI_validation_dataset, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = dataset_config['percentage_train'],
                                                                                       idx_to_use = idx_client,
                                                                                       seed = training_config['seed'],
                                                                                       preprocess_functions = preprocess_functions
                                                                                       )
    
    if training_config['print_var'] :
        print("#######################################")
        print("CLIENT")
        print("dataset_config['merge_AD_class'] ", dataset_config['merge_AD_class'])
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
    model = vgg_nets.get_vgg(model_config, return_preprocess_functions = False)
    model.set_training_mode(training_config['vgg_training_mode'])

    return client.flower_client_v1(model, MRI_train_dataset, MRI_validation_dataset, training_config).to_client()

app = ClientApp(client_fn_vgg)
