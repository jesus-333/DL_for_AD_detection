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

from . import metrics, support_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def train(training_config : dict, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
    """
    Train a model using the given configuration and dataset.

    Parameters
    ----------
    training_config : dict
        Dictionary containing the training configuration. The dictionary will be check through the function check_training_config.
        If some keys are missing a defualt value will be used or an error will be raised (depending on the key).
        The dictionary should contain the following keys : (TODO : Check if need to be updated)
        - batch_size : int
            Batch size to use for training. If missing an error will be raised.
        - lr : float
            Learning rate to use for training. If missing or it has a value <= 0 an error will be raised.
        - epochs : int
            Number of epochs to use for training. If missing or it has a value <= 0 an error will be raised.
        - optimizer_config : dict
            Dictionary with the configuration of the optimizer.
        - use_scheduler : bool
            If True, a learning rate scheduler will be used. If not specified, False will be used as default value.
        - lr_scheduler_config : dict
            Dictionary with the configuration of the selected lr scheduler. The possible keys depend from the selected scheduler (see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for a complete list).
            Scheduler currently implemented : ExponentialLR, CosineAnnealingLR.
        - device : str
            Device to use for training. If not specified, "cpu" will be used as default value.
        - epoch_to_save_model : int
            Number of epochs to use for saving the model. It must be a positive integer. If not specified, or it has a value <= 0, the model will be saved only at the end of the training.
        - path_to_save_model : str
            Path to save the model. If not specified, "model_weights" will be used as default value.
        - measure_metrics_during_training : bool
            If True, additional metrics will be computed during training (e.g. accuracy, sensitivity). If not specified, True will be used as default value.
        - seed : float
            Seed to use for training. Note that this parameter is not used inside this function. Usually it is used before in the functions to split the data. See for example in demnet_kaggle_wandb.py in scripts/training.
            Keep the seed in the training_config dictionary is still useful for reproducibility (you can save locally the dictionary or log it on wandb if you use the function wandb_train)
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
    # Check config and Dataloader creation

    # Check if the training configuration
    support_training.check_training_config(training_config)
    
    # Create Dataloaders
    train_loader = support_training.get_dataloader(train_dataset, training_config['batch_size'], training_config['dataloader_config'])
    if validation_dataset is not None : validation_loader = support_training.get_dataloader(validation_dataset, training_config['batch_size'], training_config['dataloader_config'])
    else : validation_loader = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Creation of loss function, optimizer and scheduler

    # Move model to training device
    model.to(training_config['device'])
    
    # (OPTIONAL) If resume_from_backup is True, look for a checkpoint created by a previous execution of this same training (e.g. a slurm job that was preempted and requeued).
    # If a valid checkpoint is found, it has priority over path_past_weights/path_past_optimizer (that are used only for the first launch of a continuation).
    checkpoint = None
    if training_config['resume_from_backup'] :
        checkpoint = support_training.load_latest_checkpoint(training_config['path_to_save_model'], training_config['device'])
        if checkpoint is None and training_config['print_var'] : print(f"No valid checkpoint found in {training_config['path_to_save_model']}. The training starts from epoch {training_config['start_epoch']}")

    # (OPTIONAL) Load past model weights if the training is a continuation of a past training.
    # Note that a None value is not saved in toml files, so usually the key is simply missing when no past weights are used.
    if checkpoint is not None :
        model.load_state_dict(checkpoint['model_state_dict'])
        if training_config['print_var'] : print(f"Model weights loaded from {checkpoint['path']} (epoch {checkpoint['epoch']})")
    elif training_config.get('path_past_weights') is not None :
        model_state_dict = torch.load(training_config['path_past_weights'], map_location = training_config['device'])
        model.load_state_dict(model_state_dict)
        if training_config['print_var'] : print("Model weights loaded from {}".format(training_config['path_past_weights']))

    # Create loss function
    # TODO Add option for other loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Get optimizer
    optimizer = support_training.get_optimizer(training_config['optimizer_config'], model)

    # (OPTIONAL) Load past optimizer state dict if the training is a continuation of a past training.
    if checkpoint is not None :
        if checkpoint['optimizer_state_dict'] is not None : optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif training_config.get('path_past_optimizer') is not None :
        optimizer_state_dict = torch.load(training_config['path_past_optimizer'], map_location = training_config['device']) # CHECK IF I HAVE TO ADD DEVICE
        optimizer.load_state_dict(optimizer_state_dict)
        if training_config['print_var'] : print("Optimizer state dict loaded from {}".format(training_config['path_past_optimizer']))

    # (OPTIONAL) Setup lr scheduler
    if training_config['use_scheduler'] :
        lr_scheduler = support_training.get_lr_scheduler(training_config['lr_scheduler_config'], optimizer)
        if checkpoint is not None and checkpoint['lr_scheduler_state_dict'] is not None : lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    else:
        lr_scheduler = None

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Create a folder (if not exist already) to store temporary file during training
    os.makedirs(training_config['path_to_save_model'], exist_ok = True)
    os.makedirs(training_config['path_to_save_snapshot'], exist_ok = True)

    # (OPTIONAL) If wandb is installed, tell wandb to track the model parameters
    if training_config['wandb_training'] and training_config['wandb_watch_log'] != 'none' :
        wandb.watch(model, criterion = loss_function, log = training_config['wandb_watch_log'], log_freq = training_config['wandb_watch_log_freq'], log_graph = training_config['wandb_watch_log_graph'])

    # Variable to track best losses
    best_loss_val = sys.maxsize # Best total loss for the validation data

    # Epoch from which the training starts.
    # The epoch counter is global, i.e. if the training is resumed from a checkpoint at epoch n, the training restart from epoch n + 1 and stops at training_config['epochs'].
    if checkpoint is not None :
        start_epoch   = checkpoint['epoch'] + 1
        best_loss_val = checkpoint['best_loss_val']
    else :
        start_epoch = training_config['start_epoch']
    del checkpoint # Free memory (the checkpoint contains a copy of the model and optimizer state)

    # Dictionaries used to saved information during training and load them on wandb.
    # Note that this due Dictionaries serves different from purposes.
    # The variable computed_metrics_during_training is used to save each metric at each epoch and it is returned at the end of the training.
    # The variable log_dict save the metrics of a single epoch and it is used by wandb to log the metrics. It is reset at every epoch. The reset is not hardcoded (i.e. I don't have any log_dict = {} for each iteration of the cycle)
    # But at each iteration the same key are used, so basically it the same a reset because each time the values are overwritten.
    log_dict = {}
    computed_metrics_during_training = dict()

    if training_config['print_var'] : print("Start training")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for epoch in range(start_epoch, training_config['epochs']):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (MANDATORY) Training epoch

        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss = train_epoch_function(model, train_loader, loss_function, optimizer,
                                          training_config['device'], log_dict, training_config['print_var']
                                          )

        computed_metrics_during_training.setdefault("train_loss", []).append(train_loss)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Validation epoch
        # Note that the validation is done BEFORE the backups, so the best validation loss saved in the checkpoint is always up to date.

        validation_loss = None
        if validation_loader is not None:

            validation_loss = validation_epoch_function(model, validation_loader, loss_function,
                                                        training_config['device'], log_dict,
                                                        )
            
            # Save the new BEST model if a new minimum is reach for the validation loss
            if validation_loss < best_loss_val:
                best_loss_val = validation_loss

                # Save model
                support_training.atomic_torch_save(model.state_dict(), '{}/{}'.format(training_config['path_to_save_model'], 'model_BEST.pth'))

                # Save optimizer state dict (useful if you want to continue the training from the best model)
                support_training.atomic_torch_save(optimizer.state_dict(), '{}/{}'.format(training_config['path_to_save_model'], 'optimizer_BEST.pth'))

            computed_metrics_during_training.setdefault("validation_loss", []).append(validation_loss)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Optional steps during the training
        # Note that the train loss and validation loss values are still saved in dictionary metrics

        #  Measure the various metrics related to classification (accuracy, precision etc)
        if training_config['measure_metrics_during_training'] :
            # Compute the various metrics
            train_metrics_dict = metrics.compute_metrics(model, train_loader, training_config['device'])
            if validation_loader is not None : validation_metrics_dict = metrics.compute_metrics(model, validation_loader, training_config['device'])

            # Save metrics
            for metric in train_metrics_dict :
                computed_metrics_during_training.setdefault(f"{metric}_train", []).append(train_metrics_dict[metric])
                if validation_loader is not None : computed_metrics_during_training.setdefault(f"{metric}_validation", []).append(validation_metrics_dict[metric])
        
        # Save learning rate
        if training_config['wandb_training']: log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

        #  (OPTIONAL) Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None:
            # Save the current learning rate if I load the data on wandb

            # Update scheduler
            lr_scheduler.step()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Save model (snapshots and backup)
        # Note that this is done AFTER the lr scheduler step, so the saved optimizer/scheduler state is exactly the one needed for the next epoch.

        # Save the model every n epochs (i.e. created checkpoint called model_n.pth)
        # N.b. When the variable epoch is n the model is trained for n + 1 epochs when arrive at this instructions.
        if (epoch + 1) % training_config['epoch_to_save_model'] == 0 and training_config['epoch_to_save_model'] > 0:

            # Save model
            torch.save(model.state_dict(), '{}/{}'.format(training_config['path_to_save_snapshot'], "model_{}.pth".format(epoch + 1)))

            # Save optimizer state dict (useful if you want to continue the training from a specific checkpoint)
            torch.save(optimizer.state_dict(), '{}/{}'.format(training_config['path_to_save_snapshot'], "optimizer_{}.pth".format(epoch + 1)))

        # Save the model for backup every epoch of the function is enabled.
        # Note that this is different from the checkpoint saved every n epochs because the file created will be overwritten at each epoch.
        # This could be useful if you want to hava a backup of the model during training and the model is quite large (so creating many checkpoints could be a problem)
        # If resume_from_backup is True, a full checkpoint (model, optimizer, lr scheduler, epoch and best validation loss) is also saved, so the training can be resumed from this epoch.
        # Read the docstring of the function backup_every_epoch for more information about the file created and the path where it is saved.
        if training_config['backup_model_every_epoch'] :
            training_state = dict(epoch = epoch, best_loss_val = best_loss_val) if training_config['resume_from_backup'] else None
            support_training.backup_every_epoch(training_config['path_to_save_model'], epoch, model, optimizer, lr_scheduler, training_state)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Print loss. The spaces were added to make the print more aligned and more readable.
        if training_config['print_var']:
            print("Epoch:{}".format(epoch))
            
            # Losses
            print("\t Train loss        = {}".format(train_loss))
            if validation_loader is not None: print("\t Validation loss   = {}".format(validation_loss))
            
            # Learning rate
            if lr_scheduler is not None: print("\t Learning rate     = {}".format(optimizer.param_groups[0]['lr']))
            
            # Metrics
            if training_config['measure_metrics_during_training']:
                print("\t Accuracy (TRAIN)  = {}".format(train_metrics_dict['accuracy']))
                if validation_loader is not None : print("\t Accuracy (VALID)  = {}".format(validation_metrics_dict['accuracy']))

            if training_config['debug'] :
                print_debug = getattr(model, "print_debug", None)
                if callable(print_debug):
                    model.print_debug()

            print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Log data on wandb
        if training_config['wandb_training']:
            # Update the log with the epoch losses
            # The epoch is logged explicitly so it can be used as x-axis in wandb (the wandb step is not reliable when a run is resumed several times).
            log_dict['epoch'] = epoch
            log_dict['train_loss'] = train_loss
            if validation_loader is not None : log_dict['validation_loss'] = validation_loss

            # Save the metrics in the log
            if training_config['measure_metrics_during_training']:
                support_training.update_log_dict_metrics(train_metrics_dict, log_dict, 'train')
                if validation_loader is not None: support_training.update_log_dict_metrics(validation_metrics_dict, log_dict, 'validation')

            # Add the model to the artifact
            if (epoch + 1) % training_config['epoch_to_save_model'] == 0 and training_config['log_model_artifact']:
                model_file_path = '{}/{}'.format(training_config['path_to_save_snapshot'], "model_{}.pth".format(epoch + 1))
                wandb_model_artifact.add_file(model_file_path)
                wandb.save(model_file_path)
            
            wandb.log(log_dict)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # (OPTIONAL) Stop file. If the file exists, the training stops here (after the backup and the log of the current epoch).
        if training_config['stop_file_path'] is not None and os.path.exists(training_config['stop_file_path']) :
            print(f"Stop file found at {training_config['stop_file_path']}. Training stopped at epoch {epoch}")
            break

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # End training cycle

    # Save the model at the end of the training
    if training_config['backup_model_every_epoch'] :
        # TODO Implement the removal of the backup file created during the training? Or at least the last file created because it will be the same that I create now.
        model_file_path_END = f'{training_config['path_to_save_model']}/model_END.pth'
    else :
        model_file_path_END = f'{training_config['path_to_save_model']}/model_END.pth'
    torch.save(model.state_dict(), model_file_path_END)
    
    # Save in wandb the model at the end of the training (and the best model if validation is performed)
    if training_config['wandb_training'] :
        # The if are separated because if wandb_training is False the key log_model_artifact probably is not inside the training_config dictionary
        # The check for log_model_artifact inside the if avoid to raise an error if the key is not present in the training_config dictionary
        if training_config['log_model_artifact'] :
            # Path to the model weights at the end of the training

            # Save model
            wandb_model_artifact.add_file(model_file_path_END)
            wandb.save(model_file_path_END)

            # If validation is performed, save the model with the best validation loss
            if validation_loader is not None:
                model_file_path_BEST = '{}/{}'.format(training_config['path_to_save_model'], 'model_BEST.pth')
                wandb_model_artifact.add_file(model_file_path_BEST)
                wandb.save(model_file_path_BEST)

    # Return the trained model
    return model, computed_metrics_during_training

def wandb_train(config : dict, model, train_dataset, validation_dataset = None) :
    """
    Train a model using the given configuration and dataset. This function uses wandb to log the training.
    For more information about the parameters, please refer to the train function.

    Parameters
    ----------

    config : dict
        Dictionary containing the training configuration. The dictionary should contain the following keys:
        - training_config : dict : Dictionary containing the training configuration. Read the documentation of the train function for more information
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
    if 'training_config' not in config : raise ValueError('The configuration dictionary must contain the key "training_config"')
    if 'model_config'    not in config : raise ValueError('The configuration dictionary must contain the key "model_config"')

    # Get train configuration
    training_config = config['training_config']
    entity = training_config['entity'] if 'entity' in training_config else None
    name   = training_config['name_training_run'] if 'name_training_run' in training_config else None
    notes  = training_config['notes'] if 'notes' in training_config else 'No notes in training_config'
    id     = str(training_config['seed'])
    
    # Initialize wandb
    with wandb.init(
        project = training_config['project_name'], entity = entity, name = name,
        job_type = "train", config = config, notes = notes,
        id = id, resume = 'allow'
    ) as run:

        # Check if run is already running
        # if run.state == "running" : #     raise ValueError(f"A run with the same id ({id}) is already running. Please change the seed in the training_config to have a different id")

        # For resumable trainings use the epoch (logged explicitly in train) as x-axis for all the metrics.
        # When a run is resumed, wandb continues from its last internal step, so the step is not necessarily equal to the epoch
        # (e.g. a continuation that starts from start_epoch > 0, or an epoch trained but not logged because the job was preempted between the backup and wandb.log).
        if training_config.get('resume_from_backup', False) :
            run.define_metric('epoch')
            run.define_metric('*', step_metric = 'epoch')

        # Setup artifact to save model
        model_artifact_name = training_config['model_artifact_name'] + '_trained'
        metadata = config
        
        if training_config['log_model_artifact'] :
            wandb_model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                                  description = "Trained {} model".format(training_config['model_artifact_name']),
                                                  metadata = metadata
                                                  )
        else :
            wandb_model_artifact = None
        
        # Train the model
        model, computed_metrics_during_training = train(training_config, model, train_dataset, validation_dataset, wandb_model_artifact)
        
        # Log the model artifact
        if training_config['log_model_artifact'] : run.log_artifact(wandb_model_artifact)

    return model, computed_metrics_during_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Epoch functions

def train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict = None, print_var = True) :
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
