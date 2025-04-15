"""
TODO : Implement in the future.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch

from .train_functions import validation_epoch_function, update_log_dict_metrics
from . import metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def test(training_config : dict, model, test_dataset) :

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Setup for test

    # Create dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = training_config['batch_size'], shuffle = True)

    # Move model to training device
    model.to(training_config['device'])

    # Create loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Forward pass and loss computation on the test set

    log_dict = {}
    test_loss = validation_epoch_function(model, test_loader, loss_function, 
                                                training_config['device'], log_dict,
                                                )
    
    # Save the test loss with the correct key. 
    # Note that the function validation_epoch_function() save the loss inside the dictionary with the key validation_loss
    del log_dict['validation_loss']
    log_dict['test_loss'] = test_loss

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # (OPTIONAL) Optional steps during the training
    #  Measure the various metrics related to classification (accuracy, precision etc)

    if training_config['measure_metrics_during_training'] :
        # Compute the various metrics
        test_metrics_dict = metrics.compute_metrics(model, test_loader, training_config['device'])

        # Save metrics 
        update_log_dict_metrics(test_metrics_dict, log_dict)

    return test_loss, log_dict
