"""
Update the server config toml file for the FL training SIMULATION with the fed_avg_with_wandb_tracking class.
I keep it separate from the other update scripts because this one is used specifically for federated training while the others are for "general" deep learning training

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import argparse
import numpy as np
import toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create parser
parser = argparse.ArgumentParser(description = 'Update the server configuration file with new parameters.')

# Add arguments

# Various arguments
parser.add_argument('--path_server_config'         , type = str  , default = None, help = 'Path to the toml file with the server config. Default is ./config/serve.toml')
parser.add_argument('--num_rounds'                 , type = int  , default = None, help = 'Number of rounds for the federated learning. It muse be a positive integer. If not provided an error will be raised.')
parser.add_argument('--rounds_to_save_model'       , type = int  , default = -1  , help = 'Save the model every n epochs. If a negative value (or no value) is provided, it will be set to epochs + 1, i.e. only the model at the end of training will be saved.  Default is -1.')
parser.add_argument('--num_clients'                , type = int  , default = None, help = 'Number of clients for the federated learning. It must be a positive integer. If not provided an error will be raised.')
parser.add_argument('--fraction_fit'               , type = float, default = 1   , help = 'Fraction of clients to be selected for training in each round. It must be a float between 0 and 1. Default is 1 (all clients are selected).')
parser.add_argument('--fraction_evaluate'          , type = float, default = 1   , help = 'Fraction of clients to be selected for evaluation in each round. It must be a float between 0 and 1. Default is 1 (all clients are selected). For now this parameter does not have any effect because the evaluation is perfomed inside the training function (see the train function in src/training/train_functions.py). I keep it here as a placeholder for possible future use, where the evaluation is performed in a different function than the training one.')
parser.add_argument('--clients_seed'               , nargs = '+' , default = []  , help = 'List of seeds for the clients. If not provided, the seeds will be randomly generated. If provided, the length of the list must be equal to num_clients. Default is an empty list, which means that the seeds will be randomly generated.')
parser.add_argument('--path_idx_server_data'       , type = str  , default = None, help = 'Path to the folder where the data indices are stored. Used for V2 flower app. Used only if centralized_evaluation is set to True. Default is None.')
# Boolean arguments
parser.add_argument('--keep_labels_proportion'     , action = 'store_true', default = True , help = 'If True, when data are splitted among clients, the proportion of labels of the original dataset is kept for each client. If false, the labels are randomly assigned to the clients. Default is True.')
parser.add_argument('--centralized_evaluation'     , action = 'store_true', default = False, help = 'If True, the server will perform a centralized evaluation on the whole dataset. If False, the server will not perform any evaluation. Default is True. In this the central evaluation is performed the data will be divided in num_clients + 1 parts, where the last part is used for the central evaluation. If False, the data will be divided in num_clients parts only, and no central evaluation will be performed.')
parser.add_argument('--use_on_fit_config_function' , action = 'store_true', default = False, help = 'If True, the server will receive in input the on_fit_config_fn. Note that at the moment the function is written inside the server_app.py file.')
parser.add_argument('--simulation'                 , action = 'store_true', default = False, help = 'Used as a flag to indicate if the FL training is performed in simulation mode. Note that it will not any effect on the training, but it is used only to save the information in the server config file and upload it to wandb.')
# Negate boolean arguments
parser.add_argument('--no-keep_labels_proportion'    , action = 'store_false', dest = 'keep_labels_proportion'    , help = 'If passed as an argument, the labels split will be done randomly among clients')
parser.add_argument('--no-centralized_evaluation'    , action = 'store_false', dest = 'centralized_evaluation'    , help = 'If passed as an argument, the server will not perform any centralized evaluation. The data will be divided in num_clients parts only, and no central evaluation will be performed.')
parser.add_argument('--no-use_on_fit_config_function', action = 'store_false', dest = 'use_on_fit_config_function', help = 'If passed as an argument, on_fit_config_fn will be set to None')
# Wandb settings
parser.add_argument('--project_name'               , type = str            , default = None   , help = 'Name of the wandb project. Default is None.')
parser.add_argument('--entity'                     , type = str            , default = None   , help = 'Name of the wandb entity (team) where the run will be logged. Default is None.')
parser.add_argument('--model_artifact_name'        , type = str            , default = None   , help = 'Name of the wandb model artifact. Default is None.')
parser.add_argument('--name_training_run'          , type = str            , default = None   , help = 'Name of the training run in wandb. Default is None.')
parser.add_argument('--notes'                      , type = str            , default = None   , help = 'Notes for the training run in wandb. Default is None.')
parser.add_argument('--log_freq'                   , type = int            , default = 1      , help = 'Frequency of wandb logging during training. Default is 1.')
parser.add_argument('--metrics_plot_backend'       , type = str            , default = 'wandb', help = 'Backend to use for plotting the metrics. Default is "wandb". The other option is "matplotlib". For now the code is implemented but not used for matplotlib. The wandb plot (for now) create better plots to upload in the wandb dashboard. So for now this parameter is useless and you can ignore it.')
parser.add_argument('--log_model_artifact'         , action = "store_true" , default = True   , help = 'If True, the model will be logged as a wandb artifact. If you do not want to log the model, use --no-log_model_artifact. The model will be logged every rounds_to_save_model epochs')
parser.add_argument('--debug'                      , action = "store_true" , default = False  , help = 'Used only as a flag to quickly find runs in wandb. Used to test the code. Default is False.')
parser.add_argument('--no-log_model_artifact'      , action = 'store_false', dest ='log_model_artifact', help = 'If passed, no model will be logged as a wandb artifact. Only metrics will be uploaded.')
parser.add_argument('--no-debug'                   , action = 'store_false', dest ='feature')
parser.add_argument('--metrics_to_log_from_clients', nargs = '+'           , default = ['accuracy_train', 'accuracy_validation'], help = 'List of metrics to log from clients. Default is ["accuracy_train", "accuracy_validation"]. The possible metrics to loads are the one computed from the function in src/training/metrics.py with added the suffix _train or _validation.')
# Flower CPU/GPU settings. Note that saving these parameters in the server config does not affect the training in any way. The only purpose is to have a record of the settings and upload them to wandb.
# These parameters are set when the flwr run command is executed. See for example the script train_local_FL.sh inside scripts_sh/test_code/
parser.add_argument('--num_cpus'                   , type = float, default = None, help = 'Number of CPUs for each client (i.e. the options.backend.client-resources.num-cpus argument flwr run command). Note that saving these parameters in the server config does not affect the training in any way. The only purpose is to have a record of the settings and upload them to wandb.')
parser.add_argument('--num_gpus'                   , type = float, default = None, help = 'Number of GPUs for each client (i.e. the options.backend.client-resources.num-gpus argument flwr run command). Note that saving these parameters in the server config does not affect the training in any way. The only purpose is to have a record of the settings and upload them to wandb.')
parser.add_argument('--max_cpu_allowed'            , type = int  , default = None, help = 'Maximum number of CPUs visible to the client simulation (i.e. the options.backend.init_args.num_cpus argument in the flwr run command). Note that saving these parameters in the server config does not affect the training in any way. The only purpose is to have a record of the settings and upload them to wandb.')
parser.add_argument('--max_gpu_allowed'            , type = int  , default = None, help = 'Maximum number of GPUs visible to the client simulation (i.e. the options.backend.init_args.num_gpus argument in the flwr run command). Note that saving these parameters in the server config does not affect the training in any way. The only purpose is to have a record of the settings and upload them to wandb.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

server_config = dict()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check and update FL parameters

# Path to server config
if args.path_server_config is not None :
    server_config['path_server_config'] = args.path_server_config
else :
    print('No path to server config provided. Using default value ./config/server.toml')
    server_config['path_server_config'] = './config/server.toml'

# Number of federated learning rounds
if args.num_rounds is not None :
    if args.num_rounds <= 0:
        raise ValueError(f'num_rounds must be a positive integer. Provided value: {args.num_rounds}')
    server_config['num_rounds'] = args.num_rounds
else :
    raise ValueError('num_rounds must be provided and must be a positive integer.')

# Save the model every n rounds
if args.rounds_to_save_model is not None and args.rounds_to_save_model > 0 :
    server_config['rounds_to_save_model'] = args.rounds_to_save_model
    print(f"Model will be saved every {server_config['rounds_to_save_model']} epochs.")
else :
    server_config['rounds_to_save_model'] = server_config['num_rounds'] + 1
    print("Model will be saved on the end of training.")

# Number of clients
if args.num_clients is not None :
    if args.num_clients <= 0:
        raise ValueError(f'num_clients must be a positive integer. Provided value: {args.num_clients}')
    server_config['num_clients'] = args.num_clients
else :
    raise ValueError('num_clients must be provided and must be a positive integer.')

# Fraction of clients to be selected for training in each round
if args.fraction_fit is not None :
    if args.fraction_fit < 0 or args.fraction_fit > 1:
        raise ValueError(f'fraction_fit must be a float between 0 and 1. Provided value: {args.fraction_fit}')
    server_config['fraction_fit'] = args.fraction_fit

# Fraction of clients to be selected for evaluation in each round (not used for now)
if args.fraction_evaluate is not None :
    if args.fraction_evaluate < 0 or args.fraction_evaluate > 1:
        raise ValueError(f'fraction_evaluate must be a float between 0 and 1. Provided value: {args.fraction_evaluate}')
    server_config['fraction_evaluate'] = args.fraction_evaluate

# List of seeds for the clients
if args.clients_seed is not None :
    if len(args.clients_seed) == 0 :
        print('No seeds provided for the clients. Using random seeds.')
        server_config['clients_seed'] = np.random.randint(0, 2**31 - 1, size = args.num_clients)
    elif len(args.clients_seed) != args.num_clients :
        raise ValueError(f'The length of the clients_seed list must be equal to num_clients ({args.num_clients}). Provided length: {len(args.clients_seed)}')
    else :
        server_config['clients_seed'] = [int(seed) for seed in args.clients_seed]

# Path to folder where data indices are stored (for V2 flower app)
if args.path_idx_server_data is not None :
    server_config['path_idx_server_data'] = args.path_idx_server_data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Boolean arguments
server_config['keep_labels_proportion']     = args.keep_labels_proportion
server_config['centralized_evaluation']     = args.centralized_evaluation
server_config['use_on_fit_config_function'] = args.use_on_fit_config_function
server_config['simulation']                 = args.simulation

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check and update Wandb settings

wandb_config = dict()

# Wandb settings
wandb_config['wandb_training'] = True

# Project name
if args.project_name is None :
    raise ValueError("No project name provided for wandb. Please provide a valid project name.")
wandb_config['project_name'] = args.project_name

# Entity (team)
if args.entity is None : print("No entity name provided for wandb. Using default value: None.")
wandb_config['entity'] = args.entity

# Model artifact name
if args.model_artifact_name is None : print("No model artifact name provided for wandb. Using default value: model.")
wandb_config['model_artifact_name'] = "model" if args.model_artifact_name is None else args.model_artifact_name

# Name of the training run
if args.name_training_run is None or args.name_training_run == "" : print("No name provided for the training run in wandb. Using default value: None.")
wandb_config['name_training_run'] = args.name_training_run

# Notes for the training run
if args.notes is None : print("No notes provided for the training run in wandb. Using default value: None.")
wandb_config['notes'] = args.notes

# Log frequency
if args.log_freq is not None and args.log_freq > 0 :
    wandb_config['log_freq'] = args.log_freq
else :
    wandb_config['log_freq'] = 1
    print(f"Invalid log frequency provided: {args.log_freq}. Using default value: {wandb_config['log_freq']}.")

# Metrics to log from clients
if args.metrics_to_log_from_clients is not None and len(args.metrics_to_log_from_clients) > 0:
    wandb_config['metrics_to_log_from_clients'] = args.metrics_to_log_from_clients
else :
    # TODO Change to empty list if no metrics are provided
    wandb_config['metrics_to_log_from_clients'] = ['accuracy_train', 'accuracy_validation']
    print(f"No metrics to log from clients provided. Using default value: {wandb_config['metrics_to_log_from_clients']}.")

# Log model artifact
wandb_config['log_model_artifact'] = args.log_model_artifact

# Debug flag
wandb_config['debug'] = args.debug

# Update server config with wandb settings
server_config['wandb_config'] = wandb_config

# import pprint
# pprint.pprint(server_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Flower CPU/GPU settings
wandb_config['num_cpus'] = args.num_cpus
wandb_config['num_gpus'] = args.num_gpus
wandb_config['max_cpu_allowed'] = args.max_cpu_allowed
wandb_config['max_gpu_allowed'] = args.max_gpu_allowed

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save the config

with open(server_config['path_server_config'], 'w') as f:
    toml.dump(server_config, f)
