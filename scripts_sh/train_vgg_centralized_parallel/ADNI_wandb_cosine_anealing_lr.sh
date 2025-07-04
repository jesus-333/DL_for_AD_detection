#!/bin/sh

#SBATCH --job-name="train_vgg_ADNI_wandb_cyclic_lr"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=0-01:30:00
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

n_parallel_training=2

# Save the PID of the process
pid_wait_array=()

for idx in $(1, $n_parallel_training);
do

	echo "Launch training with index ${idx}"

	# Path to library
	PATH_SRC="./"

	# Paths to config files
	PATH_CONFIG_FOLDER="./config/vgg_finetuning/"
	PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset_${idx}.toml"
	PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model_vgg_${idx}.toml"
	PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training_${idx}.toml"
	PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config_${idx}.toml"
	PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_${idx}.toml"


	python ./scripts_python/training/vgg_ADNI_wandb.py \
		--path_src="${PATH_SRC}"\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_model_config="${PATH_MODEL_CONFIG}"\
		--path_training_config="${PATH_TRAINING_CONFIG}"\
		--path_data="${PATH_DATA}"\
		--name_tensor_file="${NAME_TENSOR_FILE}" &
		pid_wait_array+=($!)
done


for i in $(1, $n_parallel_training);
do
	wait
done
