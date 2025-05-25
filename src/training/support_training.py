
"""
Support functions used for the training

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def update_log_dict_metrics(metrics_dict, log_dict, label = None):
    for key, value in metrics_dict.items() :
        if label is not None :
            log_dict['{}_{}'.format(key, label)] = value
        else :
            log_dict['{}'.format(key)] = value

def check_training_config(config : dict) :

    if 'batch_size' not in config :
        raise ValueError('The training configuration must contain the key "batch_size"')

    if 'epochs' not in config :
        raise ValueError('The training configuration must contain the key "epochs"')

    if config['epochs'] <= 0 :
        raise ValueError(f"The number of epochs must be greater than 0. Current value: {config['epochs']}")

    if 'use_scheduler' not in config :
        print('Warning: the training configuration does not contain the key "use_scheduler". Default value will be used False, i.e. no learning rate scheduler will be used')
        config['use_scheduler'] = False
    
    if config['use_scheduler'] and 'lr_scheduler_config' not in config :
        raise ValueError('The training configuration must contain the key "lr_scheduler_config" if "use_scheduler" is set to True')

    if config['use_scheduler'] : check_lr_scheduler_config(config['lr_scheduler_config'])

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# optimizer config config

def get_optimizer(optimizer_config : dict, model : torch.nn.Module) :
    """
    Create and return the optimizer based on the configuration provided.

    Parameters
    ----------
    optimizer_config : dict
        The configuration of the optimizer. The configuration must contain the key 'name' with the name of the optimizer to use. Then name must be the same of hte class in torch.optim.
        Currently implemented: Adam, AdamW, LBFGS, SGD.
        The configuration must contain the parameters required by the optimizer. For example, if the name is 'AdamW', the configuration must contain the key 'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'. Parameters like 'capturable', 'differentiable', 'foreach' are not implemented and they are set to their default value.
        If a parameter is not present in the configuration, the function will throw an error. You can check the parameters through the function check_optimizer_config. The functions will check if the parameters present in the configuration are valid and will set the missing ones to their default value.
    model : torch.nn.Module
        The model to use with the optimizer. The model must be created before calling this function. The model must be the same as the one used in the training function.
    """

    if optimizer_config['name'] == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'], betas = optimizer_config['betas'], eps = optimizer_config['eps'], weight_decay = optimizer_config['weight_decay'], decoupled_weight_decay = optimizer_config['decoupled_weight_decay'], amsgrad = optimizer_config['amsgrad'], maximize = optimizer_config['maximize'])
    elif optimizer_config['name'] == 'AdamW' :
        optimizer = torch.optim.AdamW(model.parameters(), lr = optimizer_config['lr'], betas = optimizer_config['betas'], eps = optimizer_config['eps'], weight_decay = optimizer_config['weight_decay'], amsgrad = optimizer_config['amsgrad'], maximize = optimizer_config['maximize'])
    elif optimizer_config['name'] == 'LBFGS' :
        optimizer = torch.optim.LBFGS(model.parameters(), lr = optimizer_config['lr'], max_iter = optimizer_config['max_iter'], max_eval = optimizer_config['max_eval'], tolerance_grad = optimizer_config['tolerance_grad'], tolerance_change = optimizer_config['tolerance_change'], history_size = optimizer_config['history_size'], line_search_fn = optimizer_config['line_search_fn'])
    elif optimizer_config['name'] == 'SGD' :
        optimizer = torch.optim.SGD(model.parameters(), lr = optimizer_config['lr'], momentum = optimizer_config['momentum'], dampening = optimizer_config['dampening'], weight_decay = optimizer_config['weight_decay'], nesterov = optimizer_config['nesterov'], maximize = optimizer_config['maximize'])
    else :
        raise ValueError(f'The optimizer name is not valid. Curretly implemented: Adam, AdamW, LBFGS, SGD. Value given: {optimizer_config["name"]}')

    return optimizer

def check_optimizer_config(optimizer_config : dict) :
    """
    Check the parameters of the optimizer configuration and set the default values for the missing ones.
    """

    if 'lr' not in optimizer_config :
        raise ValueError('The optimizer configuration must contain the key "lr"')

    if optimizer_config['lr'] <= 0 :
        raise ValueError(f"The learning rate must be greater than 0. Current value: {optimizer_config['lr']}")

    if optimizer_config['name'] == 'Adam' :
        if 'betas' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "betas". (0.9, 0.999) will be used as default value')
            optimizer_config['betas'] = (0.9, 0.999)
        if 'eps' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "eps". 1e-8 will be used as default value')
            optimizer_config['eps'] = 1e-8
        if 'weight_decay' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "weight_decay". 0 will be used as default value')
            optimizer_config['weight_decay'] = 0
        if 'decoupled_weight_decay' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "decoupled_weight_decay". False will be used as default value')
            optimizer_config['decoupled_weight_decay'] = False
        if 'amsgrad' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "amsgrad". False will be used as default value')
            optimizer_config['amsgrad'] = False
        if 'maximize' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "maximize". False will be used as default value')
            optimizer_config['maximize'] = False
    elif optimizer_config['name'] == 'AdamW' :
        if 'betas' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "betas". (0.9, 0.999) will be used as default value')
            optimizer_config['betas'] = (0.9, 0.999)
        if 'eps' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "eps". 1e-8 will be used as default value')
            optimizer_config['eps'] = 1e-8
        if 'weight_decay' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "weight_decay". 1e-2 will be used as default value')
            optimizer_config['weight_decay'] = 1e-2
        if 'amsgrad' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "amsgrad". False will be used as default value')
            optimizer_config['amsgrad'] = False
        if 'maximize' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "maximize". False will be used as default value')
            optimizer_config['maximize'] = False
    elif optimizer_config['name'] == 'LBFGS' :
        if 'max_iter' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "max_iter". 20 will be used as default value')
            optimizer_config['max_iter'] = 20
        if 'max_eval' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "max_eval". max_eval * 1.25 will be used as default value')
            optimizer_config['max_eval'] = optimizer_config['max_iter'] * 1.25
        if 'tolerance_grad' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "tolerance_grad". 1e-7 will be used as default value')
            optimizer_config['tolerance_grad'] = 1e-7
        if 'tolerance_change' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "tolerance_change". 1e-9 will be used as default value')
            optimizer_config['tolerance_change'] = 1e-9
        if 'history_size' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "history_size". 100 will be used as default value')
            optimizer_config['history_size'] = 100
        if 'line_search_fn' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "line_search_fn". None will be used as default value')
            optimizer_config['line_search_fn'] = None
    elif optimizer_config['name'] == 'SGD' :
        if 'momentum' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "momentum". 0 will be used as default value')
            optimizer_config['momentum'] = 0
        if 'dampening' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "dampening". 0 will be used as default value')
            optimizer_config['dampening'] = 0
        if 'weight_decay' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "weight_decay". 0 will be used as default value')
            optimizer_config['weight_decay'] = 0
        if 'nesterov' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "nesterov". False will be used as default value')
            optimizer_config['nesterov'] = False
        if 'maximize' not in optimizer_config :
            print('Warning: the optimizer configuration does not contain the key "maximize". False will be used as default value')
            optimizer_config['maximize'] = False
    else :
        raise ValueError(f'The optimizer name is not valid. Curretly implemented: Adam, AdamW, LBFGS, SGD. Value given: {optimizer_config["name"]}')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# lr scheduler config

def get_lr_scheduler(lr_scheduler_config : dict, optimizer)  :
    """
    Return the learning rate scheduler based on the configuration provided.

    Parameters
    ----------
    lr_scheduler_config : dict
        The configuration of the learning rate scheduler. The configuration must contain the key 'name' with the name of the scheduler to use. Then name must be the same of hte class in torch.optim.lr_scheduler.
        Currently implemented: ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, CyclicLR, ChainedScheduler (the chained scheduler must be composed of other schedulers already implemented in this function).
        The configuration must contain the parameters required by the scheduler. For example, if the name is 'ExponentialLR', the configuration must contain the key 'gamma' with the value of gamma to use.
        If the name is 'ChainedScheduler', the configuration must contain the key 'list_config_schedulers'. The value of this key must be a dictionary with the name of the scheduler as key and the configuration as value. The configuration of the single scheduler must be the same as the one used in this function.
        Example of config for the chained scheduler :
            lr_scheduler_config = dict(
                name = 'ChainedScheduler',
                list_config_schedulers = dict(
                    config_1 = dict(
                        name = 'CosineAnnealingLR',
                        T_max = 10,
                        eta_min = 3 * 1e-5,
                    ),
                    config_2 = dict(
                        name = 'ExponentialLR',
                        gamma = 0.954,
                    )
                )
            )
    optimizer : torch.optim.Optimizer
        The optimizer to use with the learning rate scheduler. The optimizer must be created before calling this function. The optimizer must be the same as the one used in the training function.
    """
    if lr_scheduler_config['name'] == 'ExponentialLR' :
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = lr_scheduler_config['gamma'])
    elif lr_scheduler_config['name'] == 'CosineAnnealingLR' :
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = lr_scheduler_config['T_max'], eta_min = lr_scheduler_config['eta_min'])
    elif lr_scheduler_config['name'] == 'CosineAnnealingWarmRestarts' :
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = lr_scheduler_config['T_0'], T_mult = lr_scheduler_config['T_mult'], eta_min = lr_scheduler_config['eta_min'])
    elif lr_scheduler_config['name'] == 'StepLR' :
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = lr_scheduler_config['step_size'], gamma = lr_scheduler_config['gamma'])
    elif lr_scheduler_config['name'] == 'CyclicLR' :
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = lr_scheduler_config['base_lr'], max_lr = lr_scheduler_config['max_lr'], step_size_up = lr_scheduler_config['step_size_up'], step_size_down = lr_scheduler_config['step_size_down'], mode = lr_scheduler_config['mode'], gamma = lr_scheduler_config['gamma'])
    elif lr_scheduler_config['name'] == 'ChainedScheduler' :
        schedulers_list = []
        for name_config in lr_scheduler_config['list_config_schedulers'] :
            config_single_scheduler = lr_scheduler_config['list_config_schedulers'][name_config]
            tmp_scheduler = get_lr_scheduler(config_single_scheduler, optimizer)
            schedulers_list.append(tmp_scheduler)
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers_list, optimizer)
    else :
        raise ValueError(f'The learning rate scheduler name is not valid. Curretly implemented: ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, CyclicLR. Value given: {lr_scheduler_config["name"]}')

    return lr_scheduler

def check_lr_scheduler_config(lr_scheduler_config : dict) :
    if 'name' not in lr_scheduler_config :
        raise ValueError('The learning rate scheduler configuration must contain the key "name". Possible values: ExponentialLR, CosineAnnealingLR, StepLR, ChainedScheduler')

    if lr_scheduler_config['name'] == 'ExponentialLR' :
        if 'gamma' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "gamma" if the name is ExponentialLR')
    elif lr_scheduler_config['name'] == 'CosineAnnealingLR' :
        if 'T_max' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "T_max" if the name is CosineAnnealingLR')
        if 'eta_min' not in lr_scheduler_config :
            print('Warning: the learning rate scheduler configuration does not contain the key "eta_min". 0 will be used as default value')
            lr_scheduler_config['eta_min'] = 0
    elif lr_scheduler_config['name'] == 'CosineAnnealingWarmRestarts' :
        if 'T_0' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "T_0" if the name is CosineAnnealingWarmRestarts')
        if 'T_mult' not in lr_scheduler_config :
            print('Warning: the learning rate scheduler configuration does not contain the key "T_mult". 1 will be used as default value')
            lr_scheduler_config['T_mult'] = 1
        if 'eta_min' not in lr_scheduler_config :
            print('Warning: the learning rate scheduler configuration does not contain the key "eta_min". 0 will be used as default value')
            lr_scheduler_config['eta_min'] = 0
    elif lr_scheduler_config['name'] == 'StepLR' :
        if 'step_size' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "step_size" if the name is StepLR')
        if 'gamma' not in lr_scheduler_config :
            print('Warning: the learning rate scheduler configuration does not contain the key "gamma". 0.1 will be used as default value')
            lr_scheduler_config['gamma'] = 0.1
    elif lr_scheduler_config['name'] == 'CyclicLR' :
        if 'base_lr' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "base_lr" if the name is CyclicLR')
        if 'max_lr' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "max_lr" if the name is CyclicLR')
        if 'step_size_up' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "step_size_up" if the name is CyclicLR')
        if 'step_size_down' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "step_size_down" if the name is CyclicLR')
        if 'mode' not in lr_scheduler_config :
            print('Warning: the learning rate scheduler configuration does not contain the key "mode". "triangular2" will be used as default value')
            lr_scheduler_config['mode'] = 'triangular2'
        if 'gamma' not in lr_scheduler_config :
            print('Warning: the learning rate scheduler configuration does not contain the key "gamma". 1 will be used as default value')
            lr_scheduler_config['gamma'] = 1
    elif lr_scheduler_config['name'] == 'ChainedScheduler' :
        if 'list_config_schedulers' not in lr_scheduler_config :
            raise ValueError('The learning rate scheduler configuration must contain the key "list_config_schedulers" if the name is ChainedScheduler. See chained_lr_scheduler_example.toml in /scripts/training/config/ for an example.')
        for name_config in lr_scheduler_config['list_config_schedulers'] :
            check_lr_scheduler_config(lr_scheduler_config['list_config_schedulers'][name_config])
    else :
        raise ValueError(f'The name of the lr scheduler is not valid. Possible values: ExponentialLR, CosineAnnealingLR, StepLR, ChainedScheduler. Current value: {lr_scheduler_config["name"]}')

