#!/bin/sh

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="./config/test_code/"
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

python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_CONFIG_FOLDER}lr_scheduler_cosine_anealing.toml"\
	--name="CosineAnnealingLR"\
	--T_max=10\
	--eta_min=1e-5\

python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_CONFIG_FOLDER}lr_scheduler_exp.toml"\
	--name="ExponentialLR"\
	--gamma=0.94\

python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--name="ChainedScheduler"\
	--lr_scheduler_configs_path_list "${PATH_CONFIG_FOLDER}lr_scheduler_cosine_anealing.toml" "${PATH_CONFIG_FOLDER}lr_scheduler_exp.toml"\

python ./scripts_python/training/update_dataset_config.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--merge_AD_class=2\
	--percentage_train=0.7\
	--percentage_validation=0.15\
	--percentage_test=0.15\
	--apply_rescale\
	--use_normalization\
	--no-load_data_in_memory\
	
python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=128\
	--lr=1e-3\
	--epochs=3\
	--device="cuda"\
	--epoch_to_save_model=10\
	--path_to_save_model="model_weights_ADNI"\
	--seed=-1\
	--use_scheduler\
	--measure_metrics_during_training\
	--print_var\
	--wandb_training\
	--no-debug\
	--project_name="test_code"\
	--model_artifact_name="test_artifact"\
	--log_freq=1\

python ./scripts_python/training/demnet_ADNI_wandb_V3.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"

