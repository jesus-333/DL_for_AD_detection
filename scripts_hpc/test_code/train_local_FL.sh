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

# Path to data
PATH_DATA="data/ADNI_axial_3D_z_48_size_176_int/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize___int.pt"

# Dataset settings
merge_AD_class=2
percentage_train=0.7
percentage_validation=0.15
percentage_test=0.15
rescale_factor=4095

# Training settings
batch_size=128
lr=1e-3
epochs=3
device="mps"
epoch_to_save_model=-1
path_to_save_model="model_weights_ADNI"
seed=-1

# FL settings
num_rounds=10
num_clients=5
fraction_fit=1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

hatchling build
pip install .

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Reset config files (Note that this reset only the config for the client side)

python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Update dataset config

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
	--project_name="test_code"\
	--model_artifact_name="test_artifact"\
	--log_freq=1\
	--metrics_to_log_from_clients="accuracy_train accuracy_validation"\
	--metrics_plot_backend="wandb"\
	--debug\

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
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=${batch_size}\
	--lr=${lr}\
	--epochs=${epochs}\
	--device="${device}"\
	--epoch_to_save_model=${epoch_to_save_model}\
	--path_to_save_model="${path_to_save_model}"\
	--seed=-1\
	--use_scheduler\
	--measure_metrics_during_training\
	--print_var\
	--no-wandb_training\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch FL Training

flwr run ./scripts_python/training_FL/ADNI_fedavg_with_wandb/\
	--federation-config "options.num-supernodes=${num_clients}"\
	--run-config "num-server-rounds=5 local-epochs=2 path_dataset_config=\"${PATH_DATASET_CONFIG}\" path_model_config=\"${PATH_MODEL_CONFIG}\" path_server_config=\"${PATH_SERVER_CONFIG}\" path_training_config=\"${PATH_TRAINING_CONFIG}\""\
	# --run-config "num-server-rounds=5 local-epochs=2 path_dataset_config=${PATH_DATASET_CONFIG} path_model_config=${PATH_MODEL_CONFIG} path_server_config=${PATH_SERVER_CONFIG} path_training_config=${PATH_TRAINING_CONFIG}"\
	
# [tool.flwr.federations]
# # default = "local-simulation"
