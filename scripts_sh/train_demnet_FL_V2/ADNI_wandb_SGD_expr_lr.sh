#!/bin/sh

#SBATCH --job-name="train_demnet_ADNI_wandb_exp_lr"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=0-00:30:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/train_demnet_FL_V2/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/train_demnet_FL_V2/output/other_output_%x_%j.txt

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
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model.toml"
PATH_SERVER_CONFIG="${PATH_CONFIG_FOLDER}server.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"
PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_config.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config.toml"

# Path to data
# PATH_DATA="data/ADNI_axial_3D_z_${input_channels}_size_${input_size}_int/" 
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"

# Data preparation settings
percentage_data_used_for_training=0.8
seed=42
n_repetitions=1

# Dataset settings for each client
merge_AD_class=1
percentage_train=0.9
percentage_validation=0.1
percentage_test=0
rescale_factor=4095

# possible_values_percentage_data_used_for_training = [0.5, 0.75, 0.8, 0.9, 0.95, 0.98]

# Training settings
batch_size=128
epochs=10
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights/demnet_ADNI_FL_V2/exp_lr_SGD_${SLURM_JOB_ID}"

# Optimizer config
lr=1e-3
name_optimizer='SGD'
momentum=0.9
weight_decay=1e-5
dampening=0

# Lr scheduler settings
gamma=0.94

# Information about data used for model_config
input_channels=1
input_size=176

# FL settings
num_cpus=4 # Default is 2
max_cpu_allowed=4
num_gpus=1
max_gpu_allowed=1
num_clients=4
num_rounds=20
fraction_fit=1

# Always check use_vgg_normalization_values and use_rgb_input, use_pretrained_vgg
# Remember also to check the wandb config inside the server config (e.g. the log_model_artifact parameter)
# remember also to check the training config specific for fl training (e.g. use_weights_with_lower_validation_error)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Repeate the experiment n_repetitions times
for repetition in $(seq 1 $n_repetitions); do
	echo "====================================="
	echo "REPETITION ${repetition} / ${n_repetitions}"
	echo "====================================="


	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Prepare data

	srun python ./scripts_python/data_manipulation/create_idx_files_for_federated_simulations.py\
		--path_data=${PATH_DATA}\
		--name_tensor_file=${NAME_TENSOR_FILE}\
		--path_to_save="${PATH_DATA}FL_idx/"\
		--percentage_data_used_for_training=${percentage_data_used_for_training}\
		--num_clients=${num_clients}\
		--seed=${seed}\
		--no-use_cross_fold_validation\
		--keep_labels_proportion\

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
		--path_model_config=${PATH_MODEL_CONFIG}\
		--input_channels=${input_channels}\
		--input_size=${input_size}\
		--num_classes=${num_classes}

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update dataset config. Note that this settings will be applied to each client

	srun python ./scripts_python/training/update_dataset_config.py\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_data=${PATH_DATA}\
		--name_tensor_file=${NAME_TENSOR_FILE}\
		--path_idx_folder="${PATH_DATA}FL_idx/"\
		--merge_AD_class=${merge_AD_class}\
		--percentage_train=${percentage_train}\
		--percentage_validation=${percentage_validation}\
		--percentage_test=${percentage_test}\
		--apply_rescale\
		--rescale_factor=${rescale_factor}\
		--use_normalization\
		--no-use_rgb_input\
		--load_data_in_memory\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update server config

	srun python ./scripts_python/training_FL/update_server_config.py\
		--path_server_config="${PATH_SERVER_CONFIG}"\
		--num_rounds=${num_rounds}\
		--num_clients=${num_clients}\
		--fraction_fit=${fraction_fit}\
		--fraction_evaluate=1.0\
		--path_idx_server_data="${PATH_DATA}FL_idx/"\
		--centralized_evaluation\
		--simulation\
		--project_name="demnet_training_ADNI_FL_V2"\
		--entity="alberto_zancanaro_academic"\
		--model_artifact_name="demnet_z_${input_channels}"\
		--log_model_artifact\
		--log_freq=1\
		--metrics_to_log_from_clients="accuracy_train accuracy_validation"\
		--metrics_plot_backend="wandb"\
		--debug\
		--num_cpus=${num_cpus}\
		--max_cpu_allowed=${max_cpu_allowed}\
		--num_gpus=${num_gpus}\
		--max_gpu_allowed=${max_gpu_allowed}\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Update training config (CLIENT) 
	
	# Update optimizer config
	srun python ./scripts_python/training/update_optimizer.py\
		--path_optimizer_config="${PATH_OPTIMIZER_CONFIG}"\
		--name="${name_optimizer}"\
		--lr=${lr}\
		--momentum=${momentum}\
		--weight_decay=${weight_decay}\
		--dampening=${dampening}\
		# --nestorov\

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
		--fl_training\
		--use_weights_with_lower_validation_error\
		--print_var\
		--no-wandb_training\

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Launch FL Training

	srun flwr run ./scripts_python/training_FL/ADNI_demnet_fedavg_with_wandb_V2/\
		--federation-config "options.num-supernodes=${num_clients} options.backend.client-resources.num-cpus=${num_cpus} options.backend.init_args.num_cpus=${max_cpu_allowed} options.backend.client-resources.num-gpus=${num_gpus} options.backend.init_args.num_gpus=${max_gpu_allowed}"\
		--run-config "path_dataset_config=\"${PATH_DATASET_CONFIG}\" path_model_config=\"${PATH_MODEL_CONFIG}\" path_server_config=\"${PATH_SERVER_CONFIG}\" path_training_config=\"${PATH_TRAINING_CONFIG}\""\

done # End of the for loop for repetitions

