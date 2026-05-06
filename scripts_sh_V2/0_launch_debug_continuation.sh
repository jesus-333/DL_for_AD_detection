# Similar to 0_launch_sbatch.sh but for launching a training continuation, i.e. you want to continue the training of a model that has already been trained for some epochs.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Path variables.

# Package source code path
path_src="./"

# Name of past jobs you want to continue and name of the new job you want to create.
model_name="demnet" # Modify with the name of the model you want to train, e.g. demnet, vgg, resnet.
past_job_name="train_demnet_403047872_C" # Modify with the name of the previous job, i.e. the one you want to continue.
new_job_name="train_demnet_403047872_CC" # Create a new name for the new training run.

# Path with old configs and path with past wieghts
path_folder_with_previous_config="./scripts_sh_V2/config/debug/${past_job_name}/"

# Remember that for demnet you have to set use_rgb_input to false inside dataset.toml
# Also if you have problem with NUM_CLASSES and MERGE_AD_CLASS remember to read the note you write in demnet.sh, at the end of `Prepare data` section.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Hyperparameters for the new training run.
# Note that the most of the settings of the old training run will be kept for the new training run, in order to have a coherent continuation.

# Number of epochs for the new training run. Remember that the total number of epochs will be the sum of the epochs of the past training run and the epochs of the new training run.
epochs=15

# Name used by wandb when the training run is logged.
# Modify as needed. This is just for logging purposes, it does not affect the training.
name_training_run="TEST tiny (CONTINUATION)" 

# Device used for training. Modify as needed.
device="mps"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create config file for training run.

path_dataset_config="${path_folder_with_previous_config}dataset.toml"
path_model_training_config="${path_folder_with_previous_config}training.toml"
path_optimizer_config="${path_folder_with_previous_config}optimizer.toml"
path_lr_scheduler_config="${path_folder_with_previous_config}lr_scheduler.toml"
path_model_config="${path_folder_with_previous_config}model.toml"

mkdir -p ./scripts_sh_V2/config/debug/${new_job_name}/

new_path_dataset_config="./scripts_sh_V2/config/debug/${new_job_name}/dataset.toml"
new_path_training_config="./scripts_sh_V2/config/debug/${new_job_name}/training.toml"
new_path_optimizer_config="./scripts_sh_V2/config/debug/${new_job_name}/optimizer.toml"
new_path_lr_scheduler_config="./scripts_sh_V2/config/debug/${new_job_name}/lr_scheduler".toml
new_path_model_config="./scripts_sh_V2/config/debug/${new_job_name}/model.toml"

cp ${path_dataset_config} ${new_path_dataset_config}
cp ${path_model_training_config} ${new_path_training_config}
cp ${path_optimizer_config} ${new_path_optimizer_config}
cp ${path_lr_scheduler_config} ${new_path_lr_scheduler_config}
cp ${path_model_config} ${new_path_model_config}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Update current config with information from the past training run.

# Get seed
seed=$(toml get -r "${path_folder_with_previous_config}training.toml" seed)

# Get the path where the past model weights are saved.
path_past_weights_folder=$(toml get -r "${path_folder_with_previous_config}training.toml" path_to_save_model) 
path_past_weights="${path_past_weights_folder}model_END.pth"

# Update training config with the path of the past weights and the new path to save the model weights.
python ./scripts_python/training/update_training_config.py\
	--path_training_config="${new_path_training_config}"\
	--path_optimizer_config="${path_optimizer_config}"\
	--path_lr_scheduler_config="${path_lr_scheduler_config}"\
	--path_past_weights="${path_past_weights}"\
	--epochs=${epochs}\
	--seed="${seed}"\
	--device="${device}"\
	--wandb_training\
	--name_training_run="${name_training_run}"\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Script that will be launched with sbatch. Selected based on the model name.
script_name="./scripts_sh_V2/${model_name}_cent.sh"

# Data paths and names.
path_to_idx_files="${path_data}CENT_idx_${seed}/"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submit job

sh ${script_name}\
		${path_src}\
		${new_path_dataset_config}\
		${new_path_model_config}\
		${new_path_training_config}\
		${new_path_optimizer_config}\
		${new_path_lr_scheduler_config}\
		${path_to_idx_files}\
		${seed}\
