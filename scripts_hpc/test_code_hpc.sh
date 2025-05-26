#!/bin/bash -l

#SBATCH --job-name="CPU imdb"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=batch
#SBATCH --qos=normal

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

module load ai/PyTorch
source ~/venv/DL_AD/bin/activate

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

srun python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

srun python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--name="ExponentialLR"\
	--gamma=0.9\

srun python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=64\
	--lr=0.0001\
	--epochs=1\
	--device="cuda"\
	--epoch_to_save_mode=1\
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch training script

srun python ./scripts_python/training/demnet_ADNI_wandb_V3.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"\
	

