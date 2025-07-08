#!/bin/sh

#SBATCH --job-name="debug_parallel_training"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=0-00:50:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/debug/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/debug/output/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

conda init
conda activate jesus-hpc

#conda list
#pip list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

n_parallel_training=3

# Array used to save PID for each process
pid_wait_array=()

# for idx in $(1, $n_parallel_training)
for (( idx = 1; idx <= $n_parallel_training; idx++ ))
do
	
	echo "----------------------------------------------------------"
	echo "Launch training with index ${idx}"
	echo "----------------------------------------------------------"

	# Path to library
	PATH_SRC="./"

	# Paths to config files
	PATH_CONFIG_FOLDER="./scripts_sh/train_vgg_centralized_parallel/config/"
	PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset_${idx}.toml"
	PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model_vgg_${idx}.toml"
	PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training_${idx}.toml"
	PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config_${idx}.toml"
	PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_${idx}.toml"

	# Path to data
	PATH_DATA="data/ADNI_axial_middle_slice/" 
	NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"

	python ./scripts_python/training/vgg_ADNI_wandb.py \
		--path_src="${PATH_SRC}"\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_model_config="${PATH_MODEL_CONFIG}"\
		--path_training_config="${PATH_TRAINING_CONFIG}"\
		--path_data="${PATH_DATA}"\
		--name_tensor_file="${NAME_TENSOR_FILE}" &
		pid_wait_array[${idx}]=$!
done

echo "\n"

# wait for all PIDs
for pid in ${pid_wait_array[*]} 
do
	echo "Wait training $pid"
    wait $pid
done
