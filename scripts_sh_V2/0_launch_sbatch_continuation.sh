# Similar to 0_launch_sbatch.sh but for launching a training continuation, i.e. you want to continue the training of a model that has already been trained for some epochs.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Path variables.

# Package source code path
path_src="./"

# Name of past jobs you want to continue and name of the new job you want to create.
past_job_name="train_xxx" # Modify with the name of the previous job, i.e. the one you want to continue.
new_job_name="train_xxx_continuation" # Create a new name for the new training run.

# Path with old configs and path with past wieghts
path_folder_with_previous_config="./scripts_sh_V2/config/training_hpc/${past_job_name}/"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Update current config with information from the past training run.

# Get seed
seed=$(toml get -r "${path_folder_with_previous_config}training.toml" seed)

# Get the path where the past model weights are saved.
path_past_weights=$(toml get -r "${path_folder_with_previous_config}training.toml" path_to_save_model) 

# Update training config with the path of the past weights and the new path to save the model weights.
python ./scripts_python/training/update_training_config.py\
	--path_past_weights="${path_past_weights}"\

# Remember that for demnet you have to set use_rgb_input to false inside dataset.toml
# Also if you have problem with NUM_CLASSES and MERGE_AD_CLASS remember to read the note you write in demnet.sh, at the end of `Prepare data` section.

# Data paths and names.
path_to_idx_files="${path_data}CENT_idx_${seed}/"
# path_data="data/ADNI_MRI_Normalized_middle_slice/" 
# name_tensor_file="dataset_tensor___176_resize.pt"
# percentage_data_used_for_training=0.8

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Slurm variables.

# Sbatch settings
partition="gpu" # l40s, hopper, gpu
qos="normal" # besteffort, iris-hopper, normal
mem="20G"
time="00:10:00"
output="./scripts_sh_V2/output/${model_name}_continuation/out_%x_%j.txt"
error="./scripts_sh_V2/output/${model_name}_continuation/err_%x_%j.txt"

# Script that will be launched with sbatch. Selected based on the model name.
script_name="./scripts_sh_V2/${model_name}_cent.sh"

# Job name (modify as needed)
job_name="train_${model_name}_${seed}"

if [ $partition = "l40s" ] ; then
	qos="besteffort"
elif [ $partition = "hopper" ] ; then
	qos="iris-hopper"
elif [ $partition = "gpu" ] ; then
	qos="normal"
elif [ $partition = "batch" ] ; then
	qos="normal"
else 
	echo "INVALID PARTITION"
fi

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create config file for training run.

path_dataset_config="${path_folder_with_previous_config}dataset.toml"
path_model_training_config="${path_folder_with_previous_config}training.toml"
path_optimizer_config="${path_folder_with_previous_config}optimizer.toml"
path_lr_scheduler_config="${path_folder_with_previous_config}lr_scheduler.toml"
path_model_config="${path_folder_with_previous_config}model.toml"

mkdir -p ./scripts_sh_V2/config/training_hpc/${new_job_name}/

cp ${path_dataset_config} ./scripts_sh_V2/config/training_hpc/${new_job_name}/dataset.toml
cp ${path_model_training_config} ./scripts_sh_V2/config/training_hpc/${new_job_name}/training.toml
cp ${path_optimizer_config} ./scripts_sh_V2/config/training_hpc/${new_job_name}/optimizer.toml
cp ${path_lr_scheduler_config} ./scripts_sh_V2/config/training_hpc/${new_job_name}/lr_scheduler.toml
cp ${path_model_config} ./scripts_sh_V2/config/training_hpc/${new_job_name}/model.toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submit job

sbatch  --job-name=${new_job_name}\
		--partition=${partition}\
		--qos=${qos}\
		--mem=${mem}\
		--time=${time}\
		--output=${output}\
		--error=${error}\
		${script_name}\
			${path_src}\
			./scripts_sh_V2/config/training_hpc/${new_job_name}/dataset.toml\
			./scripts_sh_V2/config/training_hpc/${new_job_name}/model.toml\
			./scripts_sh_V2/config/training_hpc/${new_job_name}/training.toml\
			./scripts_sh_V2/config/training_hpc/${new_job_name}/optimizer.toml\
			./scripts_sh_V2/config/training_hpc/${new_job_name}/lr_scheduler.toml\
			${path_to_idx_files}\
			${seed}\
