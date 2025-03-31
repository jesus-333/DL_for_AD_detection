"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np

from collections import OrderedDict
from typing import List

from ..dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def split_data_for_clients(data, percentage_split_per_client : list, seed : int = None, labels = None, keep_labels_proportion : int = False) :
    """
    Split the data (and labels if provided) among the clients according to the percentage_split_per_client.
    The number of clients is supposed to be equal to the length of percentage_split_per_client.
    If keep_labels_proportion is True, the original proportion of labels is kept for each client. E.g. if the original data has 10% of label 1 and 90% of label 0, each client will have 10% of label 1 and 90% of label 0.
    Note that keep_labels_proportion works only if there are enough samples for each label for each client.
    """
    
    # Check input parameters
    # Note that check for 0.9999999999999999 is due to the float precision
    if np.sum(percentage_split_per_client) != 1 and np.sum(percentage_split_per_client != 0.9999999999999999) : raise ValueError(f"The sum of the elements in percentage_split_list must be equal to 1. Current sum is {np.sum(percentage_split_per_client)}")
    if keep_labels_proportion and labels is None : raise ValueError("keep_labels_proportion is True but labels is None")
    
    # Get indices for each client
    if keep_labels_proportion :
        idx_list = support_dataset.get_idx_to_split_data_V2(len(data), percentage_split_per_client, seed)
    else :
        idx_list = support_dataset.get_idx_to_split_data_V3(len(data), percentage_split_per_client, seed)
    
    # Variables to store data (and labels) for each client
    data_per_client = []
    labels_per_client = [] if labels is not None else None

    # Split data (and labels) for each client
    for i in range(len(percentage_split_per_client)) :
        idx_client = idx_list[i]
        data_per_client.append(data[idx_client])
        if labels is not None : labels_per_client.append(labels[idx_client])
    
    # Return values
    if labels is not None :
        return data_per_client, labels_per_client
    else :
        return data_per_client

def split_data_for_clients_uniformly(data, n_client : int, seed : int = None, labels = None, keep_labels_proportion : int = False): 
    """
    Split the data (and labels if provided) uniformly among the clients.
    If keep_labels_proportion is True, the original proportion of labels is kept for each client. E.g. if the original data has 10% of label 1 and 90% of label 0, each client will have 10% of label 1 and 90% of label 0.
    Note that keep_labels_proportion works only if there are enough samples for each label for each client.
    """

    data_per_client = []
    labels_per_client = [] if labels is not None else None

    percentage_split_per_client = [1 / n_client] * n_client

    if labels is not None :
        data_per_client, labels_per_client = split_data_for_clients(data, percentage_split_per_client, seed, labels, keep_labels_proportion)
        return data_per_client, labels_per_client
    else :
        data_per_client = split_data_for_clients(data, percentage_split_per_client, seed)
        return data_per_client

def set_weights(model, weights: List[np.ndarray]):
    """
    sLoad the weights into the model.
    The function was copied from the Flower tutorial: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
    The only difference is the change of the function signature (from set_parameters(net, paramenters) to set_weights(model, weights))
    """
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_weights(model) -> List[np.ndarray]:
    """
    Given a torch model, return a list of numpy arrays containing the weights of the model.
    The function was copied from the Flower tutorial: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.htmla
    The only difference is the change of the function signature (from get_parameters(net) to get_weights(model))
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def check_split_correctness(original_data, original_label, data_per_client, labels_per_client) :
    """
    Check that the division of data and labels per client is correct.
    The input parameters original_data and original_label must be the same used for data and labels for the function split_data_for_clients (or split_data_for_clients_uniformly).
    The input paramenters data_per_client and labels_per_client must be the output of the function.

    Note that this function simply check if each data in data_per_client has the correct label in labels_per_client.
    Yeah, I know that this is not very useful, especially if split_data_for_clients (or split_data_for_clients_uniformly) works correctly but I add for my peace of mind.
    """
    
    # For each sample in the original data create a dictionary where the key is the sample and the value is the label
    data_to_label = dict()
    for i in range(len(original_data)) : data_to_label[original_data[i]] = original_label[i]
    
    
    for i in range(len(data_per_client)) :
        data_specific_client = data_per_client[i]
        labels_specific_client = labels_per_client[i]
        for j in range(len(data_specific_client)) :
            current_data = data_specific_client[j]
            current_label = labels_specific_client[j]

            if data_to_label[current_data] != current_label :
                raise ValueError(f"Data and label do not match for client {i} and sample {j}. Actual data is {current_data}. Actual label is {current_label} and expected label is {data_to_label[current_data]}")

    print("Everything seems correct")


