#!/bin/sh

#SBATCH --job-name="train_vgg_ADNI_wandb_cosine_anealing"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=0-00:20:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/output/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

conda init
conda activate jesus-hpc

#conda list
#pip list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="./config/vgg_finetuning/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model_vgg.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config.toml"
PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer.toml"

# Path to data
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"
# PATH_DATA="data/ADNI_axial_3D_z_48_size_176_int/" 
# NAME_TENSOR_FILE="dataset_tensor___176_resize___int.pt"
# Remember to change apply_rescale to no-apply_rescale if you do not used data saved in interger (like the middle_slice)

# Dataset settings
merge_AD_class=0
percentage_train=0.8
percentage_validation=0.1
percentage_test=0.1
rescale_factor=1

# Training settings
batch_size=128
epochs=10
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights_ADNI"
seed=-1
vgg_training_mode=0

# Optimizer config
lr=1e-4
name_optimizer='SGD'
momentum=0.9
weight_decay=1e-5
dampening=0

# Lr scheduler settings
T_0=4
T_mult=2
eta_min=1e-6

# Wandb Settings
project_name="vgg_ADNI"
name_training_run="vgg_trainin_mode_${vgg_training_mode}_${name_optimizer}_cosine_warm_lr_epochs_${epochs}_batch_${batch_size}"
name_training_run="low_lr_batch_128"

# For SGD optimizer you could add/remove the nestorov parameter
# Always check use_vgg_normalization_values and use_rgb_input, use_pretrained_vgg

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

srun python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

srun python ./scripts_python/training/update_dataset_config.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--no-filter_AD_data\
	--merge_AD_class=${merge_AD_class}\
	--percentage_train=${percentage_train}\
	--percentage_validation=${percentage_validation}\
	--percentage_test=${percentage_test}\
	--no-apply_rescale\
	--rescale_factor=${rescale_factor}\
	--use_normalization\
	--load_data_in_memory\
	--no-use_rgb_input\

srun python ./scripts_python/training/update_optimizer.py\
	--path_optimizer_config="${PATH_OPTIMIZER_CONFIG}"\
	--name="${name_optimizer}"\
	--lr=${lr}\
	--momentum=${momentum}\
	--weight_decay=${weight_decay}\
	--dampening=${dampening}\
	# --nestorov\

srun python ./scripts_python/training/update_lr_scheduler.py\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--name="CosineAnnealingWarmRestarts"\
	--T_0=${T_0}\
	--T_mult=${T_mult}\
	--eta_min=${eta_min}\
	
srun python ./scripts_python/training/update_training_config.py\
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
	--print_var\
	--vgg_training\
	--use_pretrained_vgg\
	--vgg_training_mode=${vgg_training_mode}\
	--no-use_vgg_normalization_values\
	--wandb_training\
	--project_name="${project_name}"\
	--name_training_run="${name_training_run}"\
	--model_artifact_name="${model_artifact_name}"\
	--log_freq=1\
	--no-log_model_artifact
	--no-debug\

srun python ./scripts_python/training/vgg_ADNI_wandb.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"
