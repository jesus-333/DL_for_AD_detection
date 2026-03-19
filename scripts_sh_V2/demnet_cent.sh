#!/bin/sh

#SBATCH --job-name="train_demnet_CENTRALIZED_V2"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --mem=13G
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh_V2/output/swin_transformer/std_%x_%j.txt
#SBATCH --error=./scripts_sh_V2/output/swin_transformer/err_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

echo "---------------------------------------------------"
echo $CONDA_DEFAULT_ENV
echo "---------------------------------------------------"
conda init
conda activate jesus-hpc
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
echo $CONDA_DEFAULT_ENV
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"

hatchling build
pip install .

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get arguments passed to the script.

path_src=$1
path_dataset_config=$2
path_model_config=$3
path_training_config=$4
path_optimizer_config=$5
path_lr_scheduler_config=$6
path_to_idx_file=$7
seed=$8

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Print arguments (for debugging purposes).

echo "path_src: ${path_src}"
echo "path_dataset_config: ${path_dataset_config}"
echo "path_model_config: ${path_model_config}"
echo "path_training_config: ${path_training_config}"
echo "path_optimizer_config: ${path_optimizer_config}"
echo "path_lr_scheduler_config: ${path_lr_scheduler_config}"
echo "path_to_save_idx_file: ${path_to_idx_file}"
echo "seed: ${seed}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Prepare data. 
# I can use the same script to divide the data used for FL learning because the script also create an idx fill with all the indices of all training data together
# And I use the server data as validation set

path_data=$(toml get -r ${path_dataset_config} path_data)
name_tensor_file=$(toml get -r ${path_dataset_config} name_tensor_file)
percentage_data_used_for_training=$(toml get -r ${path_dataset_config} percentage_train)

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "path_data: ${path_data}"
echo "name_tensor_file: ${name_tensor_file}"
echo "path_to_idx_files: ${path_to_idx_files}"
echo "percentage_data_used_for_training: ${percentage_data_used_for_training}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

srun python ./scripts_python/data_manipulation/create_idx_files_for_federated_simulations_2.py\
	--path_data=${path_data}\
	--name_tensor_file=${name_tensor_file}\
	--path_to_save=${path_to_idx_files}\
	--percentage_data_used_for_training=${percentage_data_used_for_training}\
	--num_clients=3\
	--seed=${seed}\
	--no-use_cross_fold_validation\
	--keep_samples_proportion\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Update training config. 
srun python ./scripts_python/training/update_training_config.py\
	--path_training_config="${path_training_config}"\
	--path_optimizer_config="${path_optimizer_config}"\
	--path_lr_scheduler_config="${path_lr_scheduler_config}"\
	--seed="${seed}"\
	--no-fl_training\
	--no-vgg_training\
	--swin_training\

srun python ./scripts_python/training/demnet_centralized_V2.py\
	--path_src="${path_src}"\
	--path_dataset_config="${path_dataset_config}"\
	--path_model_config="${path_model_config_save}"\
	--path_training_config="${path_training_config}"\
	--path_to_idx_files=${path_to_idx_file}\
