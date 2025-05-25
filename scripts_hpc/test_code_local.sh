#!/bin/sh

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="./config/demnet_wandb/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config.toml"

# Path to data
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--batch_size=64\
	--lr=0.0001\
	--epochs=100\
	--use_scheduler=True\
	--device="cuda"\
	--epoch_to_save_model=10\
	--path_to_save_model="model_weights_ADNI"\
	--measure_metrics_during_training=True\
	--print_var=True\
	--seed=-1\
	--wandb_training=True\
	--project_name="demnet_ADNI"\
	--model_artifact_name="demnet_ADNI_model"\
	--log_freq=1\
	--debug=False\

python ./scripts_python/training/demnet_ADNI_wandb_V3.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"

