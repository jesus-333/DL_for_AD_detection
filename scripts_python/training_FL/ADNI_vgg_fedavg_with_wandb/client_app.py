"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import toml
import torch

from flwr.client import ClientApp, Client
from flwr.common import Context

from addl.dataset import support_dataset_ADNI
from addl.federated import client
from src.model import download_published_model, vgg_nets

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

    # Get client ID (it is a number from 0 to n_client - 1) and load indices for the client
    client_id = context.node_config["partition-id"]
    idx_client = np.load(dataset_config['path_data'] + f'{client_id}_idx.npy')

    # (OPTIONAL) Get proprocess functions.
    if dataset_config['use_normalization'] :
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

    MRI_train_dataset, MRI_validation_dataset, _ = support_dataset_ADNI.get_dataset_V2(dataset_config, percentage_split_train_val = dataset_config['percentage_train'],
                                                                                       idx_to_use = idx_client,
                                                                                       seed = training_config['seed']
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

    return client.flower_client_v1(model, MRI_train_dataset, MRI_validation_dataset, training_config).to_client()

app = ClientApp(client_fn_demnet)
