#!/bin/sh

#SBATCH --job-name="train_demnet_CENTRALIZED_V2_ADNI_exp_lr"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=13G
#SBATCH --time=0-00:35:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/train_demnet_FL_V2/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/train_demnet_FL_V2/output/other_output_%x_%j.txt

# Works as script _1 but changes how data are divided between clients
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

# Slurm ID of the FL training run you want to use to get the config
slurm_old_id=000000

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="scripts_python/training_FL/ADNI_demnet_fedavg_with_wandb_V2/config/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset_${slurm_old_id}.toml"
PATH_MODEL_CONFIG_TEMPLATE="${PATH_CONFIG_FOLDER}template/model.toml"
PATH_MODEL_CONFIG_SAVE="${PATH_CONFIG_FOLDER}model_${slurm_old_id}.toml"
PATH_SERVER_CONFIG="${PATH_CONFIG_FOLDER}server_${slurm_old_id}.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training_${slurm_old_id}.toml"
PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_config_${slurm_old_id}.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config_${slurm_old_id}.toml"

# Path to data
# PATH_DATA="data/ADNI_axial_3D_z_${input_channels}_size_${input_size}_int/" 
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"
path_to_idx_file="${PATH_DATA}FL_idx_${slurm_old_id}/"
# N.B. The file for ADNI_middle_slice were saved with value alreay normalized between 0 and 1. 

# Data preparation settings
percentage_data_used_for_training=0.9
seed=${slurm_old_id}
n_repetitions=1

# Training settings
batch_size=192
epochs=5
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights/demnet_ADNI_FL_V2/exp_lr_SGD_${SLURM_JOB_ID}"

# Optimizer config
# CHANGE only the lr. DO NOT CHANGE the other parameters
lr=1e-3
lr=5e-4
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

# Wandb Setting
project_name="demnet_CENTRALIZED_V2_all_classes_SUBJ_div"
project_name="DEBUG_DEMNET"

# Always check use_vgg_normalization and use_rgb_input, use_pretrained_vgg
# Remember also to check the wandb config inside the server config (e.g. the log_model_artifact parameter). Current no artifcat is loaded (no-log_model_artifact passed to update_server_config)
# Remember also to check the training config specific for fl training (e.g. use_weights_with_lower_validation_error)
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Repeate the experiment n_repetitions times
for repetition in $(seq 1 $n_repetitions); do
	echo "====================================="
	echo "REPETITION ${repetition} / ${n_repetitions}"
	echo "====================================="

	if [ $repetition -eq 1 ] ; then
		num_clients=5
	elif [ $repetition -eq 2 ] ; then
		num_clients=10
	elif [ $repetition -eq 3 ] ; then
		num_clients=15
	elif [ $repetition -eq 4 ] ; then
		num_clients=20
	elif [ $repetition -eq 5 ] ; then
		num_clients=25
	elif [ $repetition -eq 6 ] ; then
		num_clients=30
	fi

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Reset config files (Note that this reset only the config for the client side)

	# srun python ./scripts_python/training/reset_config_files.py\
	# 	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	# 	--path_training_config="${PATH_TRAINING_CONFIG}"\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update training config (CLIENT) 

	echo "NUM CLASSES ${num_classes}"
	
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
		--wandb_training\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Launch CENTRALIZED Training


done # End of the for loop for repetitions

