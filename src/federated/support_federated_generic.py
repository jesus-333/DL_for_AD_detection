"""
Support functions for the FL module. It contains functions related to data management, model weights conversion and metric computation.
Note that some of this functions are written to be used outside the src libary (e.g. split_data_for_clients) while others (e.g. set_weights) are for internal use.
The server has its own dedicate support file. See support_federated_server.py

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import numpy as np
import os

from collections import OrderedDict
from typing import List

from ..dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Data related functions (e.g. split data)

def split_data_for_clients(data, percentage_split_per_client : list, seed : int = None, labels = None, keep_labels_proportion : int = False) :
    """
    Split the data (and labels if provided) among the clients according to the percentage_split_per_client.
    The number of clients is supposed to be equal to the length of percentage_split_per_client.
    If keep_labels_proportion is True, the original proportion of labels is kept for each client. E.g. if the original data has 10% of label 1 and 90% of label 0, each client will have 10% of label 1 and 90% of label 0.
    Note that keep_labels_proportion works only if there are enough samples for each label for each client.
    """
    
    # Check input parameters
    # Note that check for the number different than 1 are necessary due to floating point precision "error"
    possible_sum = [1, 0.9999999999999999, 0.9999999999999998, 1.0000000000000001, 1.0000000000000002]
    if np.sum(percentage_split_per_client) not in possible_sum : raise ValueError(f"The sum of the elements in percentage_split_list must be equal to 1. Current sum is {np.sum(percentage_split_per_client)}")
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

def split_data_for_clients_uniformly(data, num_clients : int, seed : int = None, labels = None, keep_labels_proportion : int = False) :
    """
    Split the data (and labels if provided) uniformly among the clients.
    If keep_labels_proportion is True, the original proportion of labels is kept for each client. E.g. if the original data has 10% of label 1 and 90% of label 0, each client will have 10% of label 1 and 90% of label 0.
    Note that keep_labels_proportion works only if there are enough samples for each label for each client. Also if class are highly unbalanced, and/or the number of sample for specific class is not divisible by num_clients, the split is not perfectly uniform.
    """

    data_per_client = []
    labels_per_client = [] if labels is not None else None

    percentage_split_per_client = [1 / num_clients] * num_clients

    if labels is not None :
        data_per_client, labels_per_client = split_data_for_clients(data, percentage_split_per_client, seed, labels, keep_labels_proportion)
        return data_per_client, labels_per_client
    else :
        data_per_client = split_data_for_clients(data, percentage_split_per_client, seed)
        return data_per_client

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

def load_data_and_labels(file_path_list_client : str, label_list_int_client : str) :
    """
    Load the data and the labels specified by the two paths.

    The data files is an array of length n, with n the number of samples. Each element of the array must be a path to an image.
    The labels file is an array of length n, with n the number of samples. Each element of the array must be an integer representing the class of the corresponding image.
    The reason for this is that the dataset class used path to images as input during creation.
    The choice of loading the data directly in the memory, or keep the path and load at fly during the training, is left to the user (through the parameter load_data_in_memory in the dataset creation).
    """

    # Check if element of file_path_list_client are paths to images
    for i in range(len(file_path_list_client)) :
        if not isinstance(file_path_list_client[i], str) : raise ValueError(f"Element {i} of file_path_list_client is not a string. Actual value is {file_path_list_client[i]} and type is type {type(file_path_list_client[i])}")
        if not file_path_list_client[i].endswith('.jpg') and not file_path_list_client[i].endswith('.png') and not file_path_list_client[i].endswith('.jpeg') : raise ValueError(f"Element {i} of file_path_list_client is not a valid image path. Actual value is {file_path_list_client[i]}")

    # Check if labels_client is an array of integers
    for i in range(len(label_list_int_client)) :
        # If value are saved in numpy integer isinstance will return false. More info here https://github.com/numpy/numpy/issues/2951
        # TODO write a better condition in the if
        # if not isinstance(label_list_int_client[i], int) : raise ValueError(f"Element {i} of labels_client is not an integer. Actual value is {label_list_int_client[i]} and type is {type(label_list_int_client[i])}")
        if label_list_int_client[i] < 0 : raise ValueError(f"Element {i} of labels_client is negative. Actual value is {label_list_int_client[i]}. Note that labels must be integers in the range [0, n_classes - 1], where n_classes is the number of classes in the dataset.")
    
    return file_path_list_client, label_list_int_client

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model related function

def set_weights(model, weights: List[np.ndarray]):
    """
    Load the weights into the model.
    The function was originally copied from the Flower tutorial: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
    Unfortunately it was not working due to a problem with the layers that have the num_batches_tracked.
    A modified version of the dictionary copy was therefore implemented.
    """
    # Old method
    # params_dict = zip(model.state_dict().keys(), weights)
    # state_dict = OrderedDict({k : torch.Tensor(v) for k, v in params_dict})
    
    # New method
    # With the old method ...
    state_dict = dict()
    for k, v in zip(model.state_dict().keys(), weights) :
        if 'num_batches' in k :
            state_dict[k] = model.state_dict()[k]
        else :
            state_dict[k] = torch.Tensor(v)
    state_dict = OrderedDict(state_dict)

    try :
        model.load_state_dict(state_dict, strict = True)
    except RuntimeError as e:
        print("Error loading state dict. Check if the model and the weights have the same architecture.")
        print(f"Error message: {e}")
        raise e

def get_weights(model) -> List[np.ndarray]:
    """
    Given a torch model, return a list of numpy arrays containing the weights of the model.
    The function was copied from the Flower tutorial: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.htmla
    The only difference is the change of the function signature (from get_parameters(net) to get_weights(model))
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def save_model_weights(model : torch.nn.Module, path_to_save_model : str, filename : str) :
    """
    Save the model after a training round in the path specified in path_to_save_model with the name specified by filename
    The model is saved in the form of a state_dict, which is a dictionary containing all the parameters of the model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    path_to_save_model : str
        The path where to save the model. If it does not exist, it will be created through os.makedirs.
    filename : str
        The name of the file where to save the model. The file will be saved in the path_to_save_model directory.

    Returns
    -------
    str
        The complete path where the model is saved. It is the concatenation of path_to_save_model and filename, i.e. path_to_save_model/filename.
    """
    # Create folder specified in the path
    os.makedirs(path_to_save_model, exist_ok = True)

    # Save the model
    save_path = os.path.join(path_to_save_model, filename)
    torch.save(model.state_dict(), save_path)

    return save_path

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Metric related functions

def weighted_average(metrics_per_client):
    """
    This function is loosely inspired by the function with the same name in https://github.com/adap/flower/blob/main/examples/advanced-pytorch/pytorch_example/server_app.py
    Note that this function is written to be used as argument during the creation of the server (i.e.e the strategy). More specifically it should be used as value for the argoments fit_metrics_aggregation_fn and evaluate_metrics_aggregation_fn of the strategy constructor.
    After that, when this function is called, it expects to receive as input a list in the following form :
        metrics_per_client = [..., (n_i, metrics_dict_i), ...]
    The length of the list is the number of clients, n_i is the number of training samples for that specific client and metrics_dict_i is contains the metric computed for that specific client.
    """

    weighted_metrics = dict()
    n_samples_per_client = []
    # eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
    
    # Iterate over the metrics obtained for each client
    for i in range(len(metrics_per_client)) :
        # Get number of training samples and metrics for each client
        n_samples_current_client = metrics_per_client[i][0]
        metrics_current_client   = metrics_per_client[i][1]
    
        # Save the number of samples
        n_samples_per_client.append(n_samples_current_client)
        
        # Iterate over the computed metrics
        for metric in metrics_current_client :
            # If the metric is not present in the dictionary add it
            if metric not in weighted_metrics :
                # Notes about data structure :
                # Note that in weighted_metrics dictionary for each metric, I have to save the value for each client.
                # E.g. If I have a metric called 'loss_res' and 3 clients then weighted_metrics will have the following form :
                # weighted_metrics = dict(
                #     ...
                #     loss_res = numpy_array[value_loss_res_client_1, value_loss_res_client_2, value_loss_res_client_3]
                #     ...
                # )
                # So each array must have the same length of the number of client. This is equivalent to the length of metrics_per_client.
                # Remember that metrics_per_client is a list of length equals the number of clients where each element is a dictionary with the metric of that specific client.
                weighted_metrics[metric] = np.zeros(len(metrics_per_client))
            
            # Save the current metric
            weighted_metrics[metric][i] = metrics_current_client[metric]

    # Perform weighted average
    for metric in weighted_metrics :
        metric_array = weighted_metrics[metric]
        weighted_metrics[metric] = np.average(metric_array, weights = n_samples_per_client)

    # Aggregate and return custom metric (weighted average)
    return weighted_metrics
