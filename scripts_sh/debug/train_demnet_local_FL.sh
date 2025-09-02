#!/bin/sh

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="scripts_python/training_FL/ADNI_demnet_fedavg_with_wandb/config/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model.toml"
PATH_SERVER_CONFIG="${PATH_CONFIG_FOLDER}server.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"
PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_config.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config.toml"

# Path to data
# PATH_DATA="data/ADNI_axial_3D_z_${input_channels}_size_${input_size}_int/" 
# NAME_TENSOR_FILE="dataset_tensor___176_resize___int.pt"
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"

# Dataset settings for each client
merge_AD_class=0
percentage_train=0.9
percentage_validation=0.1
percentage_test=0
rescale_factor=1

# Training settings
batch_size=128
epochs=3
device="mps"
epoch_to_save_model=-1
path_to_save_model="model_weights_ADNI/debug_FL"
seed=-1

# Optimizer config
lr=1e-3
name_optimizer='AdamW'
beta_low=0.9
beta_high=0.999
eps=1e-8
weight_decay=1e-5

# Lr scheduler settings
gamma=0.94

# Information about data used for model_config
input_channels=1
input_size=176

# FL settings
num_cpus=4 # Default is 2
max_cpu_allowed=12
num_gpus=0.3
max_gpu_allowed=1
num_clients=4
num_rounds=3
fraction_fit=1

# Always check use_vgg_normalization_values and use_rgb_input, use_pretrained_vgg
# Remember also to check the wandb config inside the server config (e.g. the log_model_artifact parameter)
# Remember also to check the training config specific for FL training (e.g. use_weights_with_lower_validation_error)
#
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

if [ $merge_AD_class -eq 0 ] ; then
	num_classes=6
elif [ $merge_AD_class -eq 1 ] ; then
	num_classes=2
elif [ $merge_AD_class -eq 2 ] ; then
	num_classes=4
else 
	echo "INVALID VALUE FOR merge_AD_class in the shell script"
	num_classes=-1
fi
echo "NUM CLASSES ${num_classes}"

python ./scripts_python/training/update_model_config_demnet.py\
	--path_model_config=${PATH_MODEL_CONFIG}\
	--input_channels=${input_channels}\
	--input_size=${input_size}\
	--num_classes=${num_classes}

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
	--no-apply_rescale\
	--rescale_factor=${rescale_factor}\
	--use_normalization\
	--load_data_in_memory\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Update server config

python ./scripts_python/training_FL/update_server_config.py\
	--path_server_config="${PATH_SERVER_CONFIG}"\
	--num_rounds=${num_rounds}\
	--n_client=${num_clients}\
	--fraction_fit=${fraction_fit}\
	--fraction_evaluate=1.0\
	--keep_labels_proportion\
	--centralized_evaluation\
	--use_on_fit_config_function\
	--project_name="test_code"\
	--entity="alberto_zancanaro_academic"\
	--model_artifact_name="test_artifact"\
	--no-log_model_artifact\
	--log_freq=1\
	--metrics_to_log_from_clients="accuracy_train accuracy_validation"\
	--metrics_plot_backend="wandb"\
	--debug\
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
	--gamma=0.9\
	
# Update training config. Note that this are the config for the local training runs
python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_optimizer_config="${PATH_OPTIMIZER_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=${batch_size}\
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
	--run-config "path_dataset_config=\"${PATH_DATASET_CONFIG}\" path_model_config=\"${PATH_MODEL_CONFIG}\" path_server_config=\"${PATH_SERVER_CONFIG}\" path_training_config=\"${PATH_TRAINING_CONFIG}\""\

# N.B. If in the future you will check flower slack or documentation and you will see the parameter local-simulation-gpu remember that it's only the name of the federation used in the flower example
# It is not a parameter that have any influences in your code
