"""
@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
import os
import sys

try :
    import wandb
except ImportError :
    print('Warning: wandb is not installed. If you want to use it, please install it using "pip install wandb"')
    print('The functionality of the code will not be affected, but you will not be able to use wandb to monitor the training')

from . import metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train(train_config : dict, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train a model using the given configuration and dataset.

    Parameters
    ----------
    train_config : dict
        Dictionary containing the training configuration. The dictionary will be check through the function check_train_config. 
        If some keys are missing a defualt value will be used or an error will be raised (depending on the key).
        The dictionary should contain the following keys:
        - batch_size : int 
            Batch size to use for training. If missing an error will be raised.
        - lr : float
            Learning rate to use for training. If missing or it has a value <= 0 an error will be raised.
        - epochs : int
            Number of epochs to use for training. If missing or it has a value <= 0 an error will be raised.
        - use_scheduler : bool
            If True, a learning rate scheduler will be used. If not specified, False will be used as default value.
        - lr_decay_rate : float
            Learning rate decay rate. This parameter is used only if use_scheduler is set to True. If not specified, 0.99 will be used as default value.
        - optimizer_weight_decay : float
            Weight decay for the AdamW optimizer. If not specified, 0.01 will be used as default value.
        - device : str
            Device to use for training. If not specified, "cpu" will be used as default value.
        - epoch_to_save_model : int
            Number of epochs to use for saving the model. It must be a positive integer. If not specified, or it has a value <= 0, the model will be saved only at the end of the training.
        - path_to_save_model : str
            Path to save the model. If not specified, "model_weights" will be used as default value.
        - measure_metrics_during_training : bool
            If True, additional metrics will be computed during training (e.g. accuracy, sensitivity). If not specified, True will be used as default value.
        - print_var : bool
            If True, additional information will be printed during training (e.g. loss, learning rate). If not specified, True will be used as default value.
        - wandb_training : bool
            If True, wandb will be used to monitor the training. If not specified, False will be used as default value. 
            If you want to track the training with wandb you should use the function wandb_train instead of this one (Note that wandb_train function will internally call this function).
        - project_name : str
            Name of the wandb project. This parameter is used only if wandb_training is set to True. If not specified, an error will be raised.
        - model_artifact_name : str
            Name of the wandb model artifact. This parameter is used only if wandb_training is set to True. If not specified, an error will be raised.
        - log_freq : int
            Frequency of logging the metrics on wandb. This parameter is used only if wandb_training is set to True. If not specified, 1 will be used as default value.
        - name_training_run : str
            Name of the wandb training run. This parameter is used only if wandb_training is set to True. If not specified, None will be used as default value.
        - notes : str
            Notes to add to the wandb training run. This parameter is used only if wandb_training is set to True. If not specified, None will be used as default value.
        - debug : bool
            This key is not used in the training functions. It is only useful if you want to quickly filter the training run in wandb. If not specified, False will be used as default value.
    model : torch.nn.Module
        Model to train
    train_dataset : torch.utils.data.Dataset 
        Dataset to use for training
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset to use for validation, by default None. If None, no validation will be performed
    wandb_model_artifact : wandb.Artifact, optional
        If wandb is installed, the artifact of the model to use for logging, by default None. If None, no logging will be performed. 
        If you want to track the training with wandb you should use the function wandb_train instead of this one (Note that wandb_train function will internally call this function).
    """
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Chek config and Dataloader creation

    # Check if the training configuration 
    check_train_config(train_config)
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
    if validation_dataset is not None :  validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
    else : validation_dataloader = None
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Creation of loss function, optimizer and scheduler

    # Move model to training device
    model.to(train_config['device'])
    
    # Create loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = train_config['lr'],
                                  weight_decay = train_config['optimizer_weight_decay']
                                  )

    # (OPTIONAL) Setup lr scheduler
    if train_config['use_scheduler'] :
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
    else:
        lr_scheduler = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Create a folder (if not exist already) to store temporary file during training
    os.makedirs(train_config['path_to_save_model'], exist_ok = True)

    # (OPTIONAL) If wandb is installed, tell wandb to track the model
    if train_config['wandb_training']: wandb.watch(model, log = "all", log_freq = train_config['log_freq'])

    # Variable to track best losses
    best_loss_val = sys.maxsize # Best total loss for the validation data

    # Dictionaries used to saved information during training and load them on wandb.
    # Note that this due dcitionaries serves different from purposes. 
    # computed_metrics_during_training is used to save each metric at each epoch and it is returned at the end of the training. 
    # log_dict save the metrics of a single epoch and it is used by wandb to log the metrics. It is reset at every epoch. The reset is not hardcoded (i.e. I don't have any log_dict = {} for each iteration of the cycle)
    # But at each iteration the same key are used so basically it the same a reset because each time the values are overwritten.
    log_dict = {}
    computed_metrics_during_training = dict()

    if train_config['print_var'] : print("Start training")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for epoch in range(train_config['epochs']):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (MANDATORY) Training epoch

        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss = train_epoch_function(model, train_loader, loss_function, optimizer, 
                                          train_config['device'], log_dict, train_config['print_var']
                                          )

        # Save the model after the epoch
        # N.b. When the variable epoch is n the model is trained for n + 1 epochs when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0 and train_config['epoch_to_save_model'] > 0:
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (OPTIONAL) Validation epoch

        if validation_loader is not None:

            validation_loss = validation_epoch_function(model, validation_loader, loss_function, 
                                                        train_config['device'], log_dict,
                                                        )
            
            # Save the new BEST model if a new minimum is reach for the validation loss
            if validation_loss < best_loss_val:
                best_loss_val = validation_loss
                torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Optional steps during the training

        #  Measure the various metrics related to classification (accuracy, precision etc)
        if train_config['measure_metrics_during_training'] :
            # Compute the various metrics
            train_metrics_dict = metrics.compute_metrics(model, train_loader, train_config['device'])
            validation_metrics_dict = metrics.compute_metrics(model, validation_loader, train_config['device'])

            # Save metrics 
            for metric in train_metrics_dict :
                if epoch == 0 : # If it is the first epoch create the list for the specific metric
                    computed_metrics_during_training[f"{metric}_train"]      = [train_metrics_dict[metric]]
                    computed_metrics_during_training[f"{metric}_validation"] = [validation_metrics_dict[metric]]
                else : # In all other cases append the metrics computed in the current epoch to the relative dictionary
                    computed_metrics_during_training[f"{metric}_train"].append(train_metrics_dict[metric])
                    computed_metrics_during_training[f"{metric}_validation"].append(validation_metrics_dict[metric])

        #  Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None:
            # Save the current learning rate if I load the data on wandb
            if train_config['wandb_training']: log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

            # Update scheduler
            lr_scheduler.step()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (OPTIONAL) Print loss
        if train_config['print_var']:
            print("Epoch:{}".format(epoch))
            print("\t Train loss        = {}".format(train_loss))
            if validation_loader is not None: print("\t Validation loss   = {}".format(validation_loss))

            if lr_scheduler is not None: print("\t Learning rate     = {}".format(optimizer.param_groups[0]['lr']))
            if train_config['measure_metrics_during_training']:
                print("\t Accuracy (TRAIN)  = {}".format(train_metrics_dict['accuracy']))
                print("\t Accuracy (VALID)  = {}".format(validation_metrics_dict['accuracy']))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (OPTIONAL) Log data on wandb
        if train_config['wandb_training']:
            # Update the log with the epoch losses
            log_dict['train_loss'] = train_loss
            log_dict['validation_loss'] = validation_loss
        
            # Save the metrics in the log 
            if train_config['measure_metrics_during_training']:
                update_log_dict_metrics(train_metrics_dict, log_dict, 'train')
                if validation_loader is not None: update_log_dict_metrics(validation_metrics_dict, log_dict, 'validation')
            
            # Add the model to the artifact
            if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
                model_file_path = '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1))
                wandb_model_artifact.add_file(model_file_path)
                wandb.save(model_file_path)
            
            wandb.log(log_dict)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # End training cycle

    # Save the model at the end of the training
    torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], "model_END.pth"))
    
    # Save in wandb the model with the best loss on validation set 
    if train_config['wandb_training'] and validation_loader is not None:
        model_file_path = '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth')
        wandb_model_artifact.add_file(model_file_path)
        wandb.save(model_file_path)

    # Return the trained model
    return model, computed_metrics_during_training

def wandb_train(config : dict, model, train_dataset, validation_dataset = None) :
    """
    Train a model using the given configuration and dataset. This function uses wandb to log the training.
    For more inforamation about the parameters, please refer to the train function.

    Parameters
    ----------

    config : dict
        Dictionary containing the training configuration. The dictionary should contain the following keys:
        - train_config : dict : Dictionary containing the training configuration. Read the documentation of the train function for more information
        - model_config : dict : Dictionary containing the model configuration. Read the documentation in the model module for more information about specific models
                                The model_config are not used during the training, but they are logged in wandb.
    model : torch.nn.Module
        Model to train
    train_dataset : torch.utils.data.Dataset
        Dataset to use for training
    validation_dataset : torch.utils.data.Dataset, optional
        Dataset to use for validation, by default None. If None, no validation will be performed 
    """
    
    # Check config
    if 'train_config' not in config : raise ValueError('The configuration dictionary must contain the key "train_config"')
    if 'model_config' not in config : raise ValueError('The configuration dictionary must contain the key "model_config"')
    
    # Get train configuration
    train_config = config['train_config']
    notes = train_config['notes'] if 'notes' in train_config else 'No notes in train_config'
    name = train_config['name_training_run'] if 'name_training_run' in train_config else None
    
    # Initialize wandb
    with wandb.init(project = train_config['project_name'], job_type = "train", config = config, notes = notes, name = name) as run:
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = config
        wandb_model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata
                                        )
        
        # Train the model
        model, computed_metrics_during_training = train(train_config, model, train_dataset, validation_dataset, wandb_model_artifact)
        
        # Log the model artifact
        run.log_artifact(wandb_model_artifact)

    return model, computed_metrics_during_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Epoch functions

def train_epoch_function(model, train_loader,loss_function, optimizer, device, log_dict = None, print_var = True) : 
    # Set the model in training mode
    model.train()

    # Variable to accumulate the loss
    train_loss = 0

    for sample_data_batch, sample_label_batch in train_loader:
        # Move data to training device
        x = sample_data_batch.to(device)
        true_label = sample_label_batch.to(device)

        # Zeros past gradients
        optimizer.zero_grad()
        
        # Networks forward pass
        pred_label = model(x)
        
        # Loss evaluation
        batch_train_loss = loss_function(pred_label, true_label)
    
        # Backward/Optimization pass
        batch_train_loss.backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss.item() * x.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss'] = float(train_loss)
    
    return train_loss

def validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict = None) : 
    # Set the model in training mode
    model.eval()

    # Variable to accumulate the loss
    validation_loss = 0
    
    # Disable gradient computation
    with torch.no_grad():

        for sample_data_batch, sample_label_batch in validation_dataloader :
            # Move data to training device
            x = sample_data_batch.to(device)
            true_label = sample_label_batch.to(device)

            # Networks forward pass
            pred_label = model(x)
            
            # Loss evaluation
            batch_train_loss = loss_function(pred_label, true_label)

            # Accumulate the loss
            validation_loss += batch_train_loss.item() * x.shape[0]

        # Compute final loss
        validation_loss = validation_loss / len(validation_dataloader.sampler)

        if log_dict is not None:
            log_dict['validation_loss'] = float(validation_loss)
        
    return validation_loss 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def check_train_config(config : dict) :

    if 'batch_size' not in config :
        raise ValueError('The training configuration must contain the key "batch_size"')

    if 'lr' not in config :
        raise ValueError('The training configuration must contain the key "lr"')

    if config['lr'] <= 0 : 
        raise ValueError(f"The learning rate must be greater than 0. Current value: {config['lr']}")

    if 'epochs' not in config :
        raise ValueError('The training configuration must contain the key "epochs"')

    if config['epochs'] <= 0 :
        raise ValueError(f"The number of epochs must be greater than 0. Current value: {config['epochs']}")

    if 'use_scheduler' not in config :
        print('Warning: the training configuration does not contain the key "use_scheduler". Default value will be used (False)')
        config['use_scheduler'] = False

    if 'lr_decay_rate' not in config :
        print('Warning: the training configuration does not contain the key "lr_decay_rate". 0.99 will be used as default value')
        print('Note that this parameter is used only if "use_scheduler" is set to True')
        config['lr_decay_rate'] = 0.99

    if 'optimizer_weight_decay' not in config :
        print('Warning: the training configuration does not contain the key "optimizer_weight_decay" for the AdamW optimizer. 0.01 will be used as default value')
        config['optimizer_weight_decay'] = 0.01

    if 'device' not in config :
        print('Warning: the training configuration does not contain the key "device". "cpu" will be used as default value')
        config['device'] = "cpu" 

    if 'path_to_save_model' not in config :
        print('Warning: the training configuration does not contain the key "path_to_save_model". "model_weights" will be used as default value')
        config['path_to_save_model'] = "model_weights"

    if 'epoch_to_save_model' not in config :
        print('Warning: the training configuration does not contain the key "epoch_to_save_model".')
        print('The model will be saved only at the end of the training')
        config['epoch_to_save_model'] = config['epochs'] + 2

    if config['epoch_to_save_model'] <= 0 :
        print('Warning: the training configuration contains the key "epoch_to_save_model" with a value <= 0.')
        print('The model will be saved only at the end of the training')
        config['epoch_to_save_model'] = config['epochs'] + 2

    if 'measure_metrics_during_training' not in config :
        print('Warning: the training configuration does not contain the key "measure_metrics_during_training". True will be used as default value')
        config['measure_metrics_during_training'] = True 

    if 'print_var' not in config :
        print('Warning: the training configuration does not contain the key "print_var". True will be used as default value')
        print('This values set to True print the metrics and the loss during training, and other information before the start of the training')
        config['print_var'] = True

    if 'wandb_training' not in config :
        print('Warning: the training configuration does not contain the key "wandb_training". False will be used as default value')
        print('If you want to use wandb to monitor the training, please set this value to True and make sure to have wandb installed')
        config['wandb_training'] = False

    if config['wandb_training'] :
        if 'project_name' not in config :
            raise ValueError('The training configuration must contain the key "project_name" if "wandb_training" is set to True')

        if 'model_artifact_name' not in config :
            raise ValueError('The training configuration must contain the key "model_artifact_name" if "wandb_training" is set to True')

        if 'log_freq' not in config :
            print('Warning: the training configuration does not contain the key "log_freq". 1 will be used as default value')
            print('This means that the metrics will be logged every epoch')
            config['log_freq'] = 1

        if 'name_training_run' not in config :
            print('Warning: the training configuration does not contain the key "name_training_run". None will be used as default value')
            print('A random name will be assigned by wandb to the training run')
            config['name_training_run'] = None

        if 'debug' not in config :
            print('Warning: the training configuration does not contain the key "debug". False will be used as default value')
            print('This key is not used in the training functions. It is only useful if you want to quickly filter the training run in wandb')
            config['debug'] = False

def update_log_dict_metrics(metrics_dict, log_dict, label = None):
    for key, value in metrics_dict.items() :
        if label is not None :
            log_dict['{}_{}'.format(key, label)] = value
        else :
            log_dict['{}'.format(key)] = value

