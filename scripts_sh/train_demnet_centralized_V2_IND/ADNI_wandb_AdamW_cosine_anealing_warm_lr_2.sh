#!/bin/sh

#SBATCH --job-name="train_demnet_CENTRALIZED_V2_ADNI_exp_lr_IND"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=13G
#SBATCH --time=0-00:30:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/train_demnet_centralized_V2_IND/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/train_demnet_centralized_V2_IND/output/other_output_%x_%j.txt

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
PATH_CONFIG_FOLDER="scripts_python/training/config/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset_${SLURM_JOB_ID}.toml"
PATH_MODEL_CONFIG_TEMPLATE="${PATH_CONFIG_FOLDER}demnet_template/model.toml"
PATH_MODEL_CONFIG_SAVE="${PATH_CONFIG_FOLDER}model_${SLURM_JOB_ID}.toml"
PATH_SERVER_CONFIG="${PATH_CONFIG_FOLDER}server_${SLURM_JOB_ID}.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training_${SLURM_JOB_ID}.toml"
PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_config_${SLURM_JOB_ID}.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config_${SLURM_JOB_ID}.toml"

# Path to data
# PATH_DATA="data/ADNI_axial_3D_z_${input_channels}_size_${input_size}_int/" 
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"
path_to_save_idx_file="${PATH_DATA}CENT_idx_${SLURM_JOB_ID}/"
# N.B. The file for ADNI_middle_slice were saved with value alreay normalized between 0 and 1. 

seed=${SLURM_JOB_ID}

# Data preparation settings
percentage_data_used_for_training=0.9
n_repetitions=1
merge_AD_class=0
rescale_factor=4095
# N.b. near the call for update_dataset_config, I add some extra notes about how dataset are created.

# Training settings
batch_size=192
epochs=100
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights/demnet_ADNI_FL_V2/exp_lr_SGD_${SLURM_JOB_ID}"

# Optimizer config
lr=1e-3
# lr=5e-4
name_optimizer='AdamW'
beta_low=0.9
beta_high=0.999
eps=1e-8
weight_decay=1e-5

# Lr scheduler settings
T_0=4
T_mult=2
eta_min=1e-6

# Information about data used for model_config
input_channels=1
input_size=176

# Wandb Setting
project_name="demnet_CENTRALIZED_V2_all_classes_SUBJ_div"
project_name="DEBUG_DEMNET_CENT"

# Always check use_vgg_normalization and use_rgb_input, use_pretrained_vgg
# Remember also to check the wandb config inside the server config (e.g. the log_model_artifact parameter). Current no artifcat is loaded (no-log_model_artifact passed to update_server_config)
# Remember also to check the training config specific for fl training (e.g. use_weights_with_lower_validation_error)
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Repeate the experiment n_repetitions times
for repetition in $(seq 1 $n_repetitions); do
	echo "====================================="
	echo "REPETITION ${repetition} / ${n_repetitions}"
	echo "====================================="

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Reset config files (Note that this reset only the config for the client side)

	srun python ./scripts_python/training/reset_config_files.py\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_training_config="${PATH_TRAINING_CONFIG}"\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

	srun python ./scripts_python/training/update_model_config_demnet.py\
		--path_save=${PATH_MODEL_CONFIG_SAVE}\
		--path_template=${PATH_MODEL_CONFIG_TEMPLATE}\
		--input_channels=${input_channels}\
		--input_size=${input_size}\
		--num_classes=${num_classes}


	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Prepare data. 
	# I can use the same script to divide the data used for FL learning because the script also create an idx fill with all the indices of all training data together
	# And I use the server data as validation set

	srun python ./scripts_python/data_manipulation/create_idx_files_for_federated_simulations_2.py\
		--path_data=${PATH_DATA}\
		--name_tensor_file=${NAME_TENSOR_FILE}\
		--path_to_save=${path_to_save_idx_file}\
		--percentage_data_used_for_training=${percentage_data_used_for_training}\
		--num_clients=3\
		--seed=${seed}\
		--no-use_cross_fold_validation\
		--keep_samples_proportion\


	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update dataset config
	# Note that in this case I set percentage_train = 1 and the other percentages to 0 but this value are not used for the training.
	# The training script `training/demnet_centralized_V2.py` also received in input the folder with the idx file to use directly for training and validation and this are created before by the `create_idx_files_for_federated_simulations_2.py` script. 
	# `percentage_train`, `percentage_validation` and `percentage_test` were used in previous versions.
	# To get the data I used the function `get_dataset_V2`. One of its input parameter is called `percentage_split_train_val` and determines the percentage of data used for training and for validation.
	# If not passed (like in the training script) it is automatically set to 1 and only the training set is returned.
	# The function `get_dataset_V2` can also receive in input a list of indices (i.e. keep only the data corresponding to the indices). I used this feature to create the train and validation set.

	srun python ./scripts_python/training/update_dataset_config.py\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_data=${PATH_DATA}\
		--name_tensor_file=${NAME_TENSOR_FILE}\
		--path_idx_folder=${path_to_save_idx_file}\
		--merge_AD_class=${merge_AD_class}\
		--percentage_train=1\
		--percentage_validation=0\
		--percentage_test=0\
		--no-apply_rescale\
		--rescale_factor=${rescale_factor}\
		--use_normalization\
		--no-use_rgb_input\
		--load_data_in_memory\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update training config

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
		--name="CosineAnnealingWarmRestarts"\
		--T_0=${T_0}\
		--T_mult=${T_mult}\
		--eta_min=${eta_min}\
		
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
		--use_scheduler\
		--measure_metrics_during_training\
		--wandb_training\
		--project_name=${project_name}\
		--entity="alberto_zancanaro_academic"\
		--model_artifact_name="demnet_z_${input_channels}"\
		--name_training_run="Cent Training AdamW IND"\
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

