#!/bin/bash -l

#SBATCH --job-name="train_demnet_ADNI_wandb_cyclic_lr"
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem=55G
#SBATCH --time=0-00:47:00
#SBATCH --qos=normal
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_hpc/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_hpc/output/other_output_%x_%j.txt

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

srun python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

srun python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--name="CyclicLR"\
	--base_lr=1e-6\
	--max_lr=1e-4\
	--step_size_up=2\
	--step_size_down=8\
	--mode="exp_range"\
	--gamma=0.99\

srun python ./scripts_python/training/update_dataset_config.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--merge_AD_class=1\
	--percentage_train=0.8\
	--percentage_validation=0.1\
	--percentage_test=0.1\
	--apply_rescale\
	--rescale_factor=4095\
	--use_normalization\
	--no-load_data_in_memory\

srun python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=128\
	--lr=1e-4\
	--epochs=100\
	--device="cuda"\
	--epoch_to_save_model=-1\
	--path_to_save_model="model_weights_ADNI"\
	--seed=-1\
	--use_scheduler\
	--measure_metrics_during_training\
	--print_var\
	--wandb_training\
	--no-debug\
	--project_name="demnet_training_ADNI_2.5D"\
	--model_artifact_name="demnet_z_48"\
	--log_freq=1\

# P.S. CHECK ALWAYS LR, EPOCH, EPOCH_TO_SAVE_MODEL
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch training script

srun python ./scripts_python/training/demnet_ADNI_wandb_V3.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"\
	

