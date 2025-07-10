#!/bin/sh

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="scripts_python/training_FL/ADNI_fedavg_with_wandb/config/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model.toml"
PATH_SERVER_CONFIG="${PATH_CONFIG_FOLDER}server.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config.toml"

# Information about data
input_channels=48
input_size=176

# Path to data
PATH_DATA="data/ADNI_axial_3D_z_${input_channels}_size_${input_size}_int/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize___int.pt"

# Dataset settings for each client
merge_AD_class=1
percentage_train=0.9
percentage_validation=0.1
percentage_test=0
rescale_factor=4095

# Lr scheduler settings
gamma=0.92

# Training settings (single client)
batch_size=128
lr=1e-3
epochs=10
device="mps"
epoch_to_save_model=-1
path_to_save_model="model_weights_ADNI"
seed=-1

# FL settings
num_rounds=50
num_clients=6
fraction_fit=1
num_cpus=3 # Default is 2
max_cpu_allowed=3
num_gpus=0
max_gpu_allowed=0

# wandb settings
project_name="demnet_training_ADNI_FL"
model_artifact_name="demnet_z_${input_channels}"
name_training_run="lr_exp_gamma_rounds_${num_rounds}_clients_${num_clients}_epochs_${epochs}_batch_${batch_size}"
log_freq=1

# Always check use_vgg_normalization_values and use_rgb_input, use_pretrained_vgg
# Remember also to check the wandb config inside the server config (e.g. the log_model_artifact parameter)
# remember also to check the training config specific for fl training (e.g. use_weights_with_lower_validation_error)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

hatchling build
pip install .

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Reset config files (Note that this reset only the config for the client side)

python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Update model config

python ./scripts_python/training/update_model_config_demnet.py\
	--path_model_config=${PATH_MODEL_CONFIG}\
	--input_channels=${input_channels}\
	--input_size=${input_size}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Update dataset config. Note that this settings will be applied to each client

python ./scripts_python/training/update_dataset_config.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_data=${PATH_DATA}\
	--name_tensor_file=${NAME_TENSOR_FILE}\
	--merge_AD_class=${merge_AD_class}\
	--percentage_train=${percentage_train}\
	--percentage_validation=${percentage_validation}\
	--percentage_test=${percentage_test}\
	--apply_rescale\
	--rescale_factor=${rescale_factor}\
	--use_normalization\
	--no-load_data_in_memory\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Update server config

python ./scripts_python/training_FL/update_server_config.py\
	--path_server_config="${PATH_SERVER_CONFIG}"\
	--num_rounds=${num_rounds}\
	--n_client=${num_clients}\
	--fraction_fit=${fraction_fit}\
	--fraction_evaluate=1.0\
	--keep_labels_proportion\
	--no-centralized_evaluation\
	--project_name="demnet_training_ADNI_FL"\
	--name_training_run="${name_training_run}"\
	--model_artifact_name="demnet_z_${input_channels}"\
	--log_freq=1\
	--metrics_to_log_from_clients="accuracy_train accuracy_validation"\
	--metrics_plot_backend="wandb"\
	--no-debug\
	--num_cpus=${num_cpus}\
	--max_cpu_allowed=${max_cpu_allowed}\
	--num_gpus=${num_gpus}\
	--max_gpu_allowed=${max_gpu_allowed}\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Training config

# Update learning rate scheduler config
python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--name="ExponentialLR"\
	--gamma=${gamma}\
	
# Update training config. Note that this are the config for the local training runs
python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=${batch_size}\
	--lr=${lr}\
	--epochs=${epochs}\
	--device="${device}"\
	--epoch_to_save_model=${epoch_to_save_model}\
	--path_to_save_model="${path_to_save_model}"\
	--seed=${seed}\
	--use_scheduler\
	--measure_metrics_during_training\
	--fl_training\
	--use_weights_with_lower_validation_error\
	--print_var\
	--no-wandb_training\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch FL Training

flwr run ./scripts_python/training_FL/ADNI_demnet_fedavg_with_wandb/\
	--federation-config "options.num-supernodes=${num_clients} options.backend.client-resources.num-cpus=${num_cpus} options.backend.init_args.num_cpus=${max_cpu_allowed} options.backend.client-resources.num-gpus=${num_gpus} options.backend.init_args.num_gpus=${max_gpu_allowed}"\
	--run-config "num-server-rounds=5 local-epochs=2 path_dataset_config=\"${PATH_DATASET_CONFIG}\" path_model_config=\"${PATH_MODEL_CONFIG}\" path_server_config=\"${PATH_SERVER_CONFIG}\" path_training_config=\"${PATH_TRAINING_CONFIG}\""\

