#!/bin/bash -l

#SBATCH --job-name="train_demnet_ADNI_wandb_chained_lr"
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem=57G
#SBATCH --time=0-01:30:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/output/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

conda activate jesus-hpc

#conda list
#pip list

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
PATH_DATA="data/ADNI_axial_3D_z_48_size_176_int/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize___int.pt"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# (OPTIONAL) Reset config
srun python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

# Dataset config
srun python ./scripts_python/training/update_dataset_config.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--merge_AD_class=2\
	--percentage_train=0.8\
	--percentage_validation=0.1\
	--percentage_test=0.1\
	--apply_rescale\
	--rescale_factor=4095\
	--use_normalization\
	--no-load_data_in_memory\

# Training config
srun python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=512\
	--lr=1e-6\
	--epochs=200\
	--device="cuda"\
	--epoch_to_save_model=-1\
	--path_to_save_model="model_weights_ADNI"\
	--seed=-1\
	--no-use_scheduler\
	--measure_metrics_during_training\
	--print_var\
	--wandb_training\
	--no-debug\
	--project_name="demnet_training_ADNI_2.5D"\
	--model_artifact_name="demnet_z_48"\
	--log_freq=1\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch training script

srun python ./scripts_python/training/demnet_ADNI_wandb_V3.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"\
	

