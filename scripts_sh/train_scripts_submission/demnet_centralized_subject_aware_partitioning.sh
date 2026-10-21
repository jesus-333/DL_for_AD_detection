#!/bin/sh

#SBATCH --job-name="train_demnet_CENTRALIZED_subject_aware_split"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=13G
#SBATCH --time=0-02:00:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/train_scripts_submission/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/train_scripts_submission/output/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Centralized counterpart of demnet_FL_subject_aware_partitioning.sh
# Data split: subject-aware split (all the samples of a subject are either in training or in validation)
# This training used
# - AdamW optimizer
# - ExponentialLR
# Differently from the old centralized V2 scripts, this script does not reuse the config files of a previous FL run. It creates all the configs from scratch.
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# set -euo pipefail

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

echo "---------------------------------------------------"
echo $CONDA_DEFAULT_ENV
echo "---------------------------------------------------"
conda init
conda activate jesus-hpc
conda init
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
echo $CONDA_DEFAULT_ENV
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"

hatchling build
pip install .

#conda list
pip list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="scripts_python/training_FL/ADNI_demnet_fedavg_with_wandb_V2/config/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset_${SLURM_JOB_ID}.toml"
PATH_MODEL_CONFIG_TEMPLATE="${PATH_CONFIG_FOLDER}template/model.toml"
PATH_MODEL_CONFIG_SAVE="${PATH_CONFIG_FOLDER}model_${SLURM_JOB_ID}.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training_${SLURM_JOB_ID}.toml"
PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_config_${SLURM_JOB_ID}.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config_${SLURM_JOB_ID}.toml"

# Path to data
PATH_DATA="data/ADNI_MRI_Normalized_middle_slice/"  # Note that the normalized dataset has only 3 classes : AD, CN and MCI
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"
path_to_save_idx_file="${PATH_DATA}CENT_idx_${SLURM_JOB_ID}/"
# N.B. The file should be already saved with value alreay normalized between 0 and 1. 

# Data preparation settings
percentage_data_used_for_training=0.9
seed=${SLURM_JOB_ID}
n_repetitions=5

# Dataset settings
merge_AD_class=0
percentage_train=0.85
percentage_validation=0.15
percentage_test=0
rescale_factor=4095

# Training settings
batch_size=192
epochs=100
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights/demnet_ADNI_CENTRALIZED/subject_aware_split_${SLURM_JOB_ID}"

# Optimizer config
lr=1e-3
name_optimizer='AdamW'
beta_low=0.9
beta_high=0.999
eps=1e-8
weight_decay=1e-5

# Lr scheduler settings
gamma=0.95

# Information about data used for model_config
input_channels=1
input_size=176

# Wandb Setting
project_name="demnet_centralized_subject_aware_split_PAPER"
name_training_run="centralized"

# Always check use_rgb_input
# Remember also to check the wandb settings (e.g. the log_model_artifact parameter). Currently no artifact is logged.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Repeate the experiment n_repetitions times
for repetition in $(seq 1 $n_repetitions); do
	echo "====================================="
	echo "REPETITION ${repetition} / ${n_repetitions}"
	echo "====================================="

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Prepare data
	# The script creates the idx files used for FL simulations. Here we use a single "client" so that train_idx_all.npy contains all the training data.

	srun python ./scripts_python/data_manipulation/create_idx_files_for_federated_simulations_2.py\
		--path_data=${PATH_DATA}\
		--name_tensor_file=${NAME_TENSOR_FILE}\
		--path_to_save=${path_to_save_idx_file}\
		--percentage_data_used_for_training=${percentage_data_used_for_training}\
		--num_clients=1\
		--seed=${seed}\
		--no-use_cross_fold_validation\
		--keep_samples_proportion\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Reset config files

	srun python ./scripts_python/training/reset_config_files.py\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_training_config="${PATH_TRAINING_CONFIG}"\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update model config

	num_classes=3
	echo "NUM CLASSES ${num_classes}"

	srun python ./scripts_python/training/update_model_config_demnet.py\
		--path_save=${PATH_MODEL_CONFIG_SAVE}\
		--path_template=${PATH_MODEL_CONFIG_TEMPLATE}\
		--input_channels=${input_channels}\
		--input_size=${input_size}\
		--num_classes=${num_classes}

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update dataset config

	srun python ./scripts_python/training/update_dataset_config.py\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_data=${PATH_DATA}\
		--name_tensor_file=${NAME_TENSOR_FILE}\
		--path_idx_folder=${path_to_save_idx_file}\
		--merge_AD_class=${merge_AD_class}\
		--percentage_train=${percentage_train}\
		--percentage_validation=${percentage_validation}\
		--percentage_test=${percentage_test}\
		--no-apply_rescale\
		--rescale_factor=${rescale_factor}\
		--use_normalization\
		--no-use_rgb_input\
		--load_data_in_memory\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update training config

	# Update optimizer config
	srun python ./scripts_python/training/update_optimizer.py\
		--path_optimizer_config="${PATH_OPTIMIZER_CONFIG}"\
		--name="${name_optimizer}"\
		--lr=${lr}\
		--betas ${beta_low} $beta_high\
		--eps=${eps}\
		--weight_decay=${weight_decay}\
		--no-amsgrad\
		--no-maximize\

	# Update learning rate scheduler config
	srun python ./scripts_python/training/update_lr_scheduler.py\
		--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
		--name="ExponentialLR"\
		--gamma=${gamma}\

	# Update training config. 
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
		--wandb_training\
		--project_name=${project_name}\
		--entity="alberto_zancanaro_academic"\
		--model_artifact_name="demnet_z_${input_channels}"\
		--name_training_run="${name_training_run}"\
		--no-log_model_artifact\
		--log_freq=1\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Launch CENTRALIZED Training

	srun python ./scripts_python/training/demnet_centralized_V2.py\
		--path_src="${PATH_SRC}"\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_model_config="${PATH_MODEL_CONFIG_SAVE}"\
		--path_training_config="${PATH_TRAINING_CONFIG}"\
		--path_to_idx_files=${path_to_save_idx_file}\

	echo "END Repetition"

done # End of the for loop for repetitions
