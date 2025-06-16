"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import pandas as pd
import toml
import torch
import torchvision

from flwr.client import ClientApp, Client
from flwr.common import Context

from addl.dataset import dataset, support_dataset, support_dataset_ADNI
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
    # dataset_config  = toml.load(context.node_config['dataset_config_path'])
    # model_config    = toml.load(context.node_config["path_model_config"])
    # training_config = toml.load(context.node_config['training_config_path'])
    
    # Set the seed
    # NOT USED NOW. Since I launch all the training from the sh script, before the flwr command I called the update_trainig.py script that already set the seed in the toml file
    # training_config['seed'] = context.run_config["seed"] if 'seed' in context.run_config else np.random.randint(0, 1e9)

    # Get dataset info
    dataset_tensor_file_name = dataset_config['name_tensor_file']
    path_to_data = dataset_config['path_data']
    dataset_info = pd.read_csv(f'{path_to_data}dataset_info.csv')
    labels_int = dataset_info['labels_int'].to_numpy()
    labels_str = dataset_info['labels_str'].to_numpy()

    # Get client ID (it is a number from 0 to n_client - 1) and load indices for the client
    client_id = context.node_config["partition-id"]
    idx_client = np.load(dataset_config['path_data'] + f'{client_id}_idx.npy')

    # Get data and labels for the specific clients
    data = torch.load(f'{path_to_data}{dataset_tensor_file_name}', mmap = True)
    data_client = data[idx_client]
    labels_int_client = labels_int[idx_client]

    # Delete old data
    del data
    
    # Get idx to split the  data in train/validation/test
    percentage_split_train_val = dataset_config['percentage_train']
    idx_list = support_dataset.get_idx_to_split_data_V3(labels_int_client, [percentage_split_train_val, 1 - percentage_split_train_val], training_config['seed'])
    idx_train, idx_validation = idx_list
    
    # (OPTIONAL) Create function to normalize the data
    if dataset_config['use_normalization'] :
        # Load procumputed dataset mean and std
        # Note that to normalize the data I still used the global mean/std
        # TODO add an option to use local mean/std
        mean_dataset = torch.load(f'{path_to_data}dataset_mean.pt')
        std_dataset  = torch.load(f'{path_to_data}dataset_std.pt')

        # Create normalization function
        preprocess_functions  = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean = mean_dataset, std = std_dataset)])
    else :
        preprocess_functions = None

    # (OPTIONAL) Merge AD classes
    labels_int = support_dataset_ADNI.merge_AD_class_function(labels_int, labels_str, dataset_config['merge_AD_class'])

    # Split data in train/validation/test
    if dataset_config['apply_rescale'] :
        MRI_train_dataset      = dataset.MRI_dataset(data_client[idx_train] / dataset_config['rescale_factor']     , labels_int_client[idx_train]     , preprocess_functions = preprocess_functions)
        MRI_validation_dataset = dataset.MRI_dataset(data_client[idx_validation] / dataset_config['rescale_factor'], labels_int_client[idx_validation], preprocess_functions = preprocess_functions)
    else :
        MRI_train_dataset      = dataset.MRI_dataset(data_client[idx_train]     , labels_int_client[idx_train]     , preprocess_functions = preprocess_functions)
        MRI_validation_dataset = dataset.MRI_dataset(data_client[idx_validation], labels_int_client[idx_validation], preprocess_functions = preprocess_functions)

    # Select training device
    if torch.cuda.is_available() :
        device = torch.device("cuda")
        print("\nCUDA backend in use")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\nmps backend (apple metal) in use")
    else:
        device = torch.device("cpu")
        print("\nNo backend in use. Device set to cpu")
    training_config['device'] = device

    # (OPTIONAL) Move dataset to device
    if dataset_config['load_data_in_memory'] :
        MRI_train_dataset.move_data_and_labels_to_device(device)
        MRI_validation_dataset.move_data_and_labels_to_device(device)
        # MRI_test_dataset.move_data_and_labels_to_device(device)
    
    # Add client id to dictionary (so ti can be saved inside the class)
    training_config['client_id'] = client_id

    # Load model
    model_config['input_channels'] = data_client.shape[1]
    model_config['num_classes'] = len(np.unique(labels_int))
    model = demnet.demnet(model_config)
    
    # Free memory
    del data_client

    return client.flower_client_v1(model, MRI_train_dataset, MRI_validation_dataset, training_config).to_client()

app = ClientApp(client_fn_demnet)
