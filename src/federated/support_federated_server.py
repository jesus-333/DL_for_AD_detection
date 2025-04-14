"""
Support functions for the FL module. It contains functions specifically written for server/strategy

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


def extract_metric_from_log_dict(log_dict : dict) -> (list, list):
    """
    Since (at the time of writing) flower does not allow to save entire array/list inside the log dict I have to save each epoch separately, i.e. with a different entry in the dictionary.
    With this method I will merge all the entry for each specic key and reconvert them to numpy array.

    Args:
        log_dict (dict): Dictionary containing the logs from the server. It must be in the format produced by the function convert_training_metrics_for_upload (see implmentation in the client.py).

    Returns:
        metrics_values_list (list): List of numpy arrays containing the values of the metrics for each epoch. Each element of the list corresponds to a different metric.
        metrics_name_list (list): List of strings containing the names of the metrics. Each element is the name of metrics corresponding to the same index in metrics_values_list.
    """
    
    training_epochs = np.arange(log_dict['epochs'])
    
    metrics_values_list = []
    metrics_name_list = []

    # Get the name of the metrics
    for metric_name in log_dict :
        if '0' in metric_name : 
            metrics_name_list.append(metric_name.split(":")[0])
    
    # Iterate over possible metrics
    for i in range(len(metrics_name_list)) :
        metric_name = metrics_name_list[i]

        # Create list for the specific metric
        metrics_values_list.append([])

        # Iterate over training epoch
        for j in range(len(training_epochs)) :
            # Get epoch and metric for the epoch
            current_epoch = training_epochs[j]
            metric_for_current_epoch = log_dict[f'{metric_name}:{current_epoch}']

            # Save metric
            metrics_values_list[i].append(metric_for_current_epoch)

        # Convert list to numpy array
        metrics_values_list[i] = np.asarray(metrics_values_list[i])
    
    return metrics_values_list, metrics_name_list

