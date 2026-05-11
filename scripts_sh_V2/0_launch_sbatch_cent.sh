set -e

# Set seed with a random positive integer (+ 1 to avoid seed=0)
seed=($RANDOM + 1)
seed=($(((RANDOM << 15) | RANDOM )) + 1)

echo "SEED = ${seed}"
echo "====================================="

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Path variables.

# Package source code path
path_src="./"

# Config files paths.
model_name="vit"
optimizer="AdamW"
lr_scheduler="CosineAnnealingWarmRestarts"

# Remember that for demnet you have to set use_rgb_input to false inside dataset.toml
# Also if you have problem with NUM_CLASSES and MERGE_AD_CLASS remember to read the note you write in demnet.sh, at the end of `Prepare data` section.

# Possible optimizer AdamW, SGD
# Possible lr lr_scheduler ExponentialLR, CosineAnnealingWarmRestarts
# IF you do not want to use any LR scheduler you have to add the optional flag --no-use_scheduler\ directly inside the script.

# Data paths and names.
# Remember that to modify the dataset you have ALSO to modify the data path inside the dataset.toml
path_data="data/ADNI_MRI_Normalized_middle_slice/" 
name_tensor_file="dataset_tensor___176_resize.pt"
path_to_idx_files="${path_data}CENT_idx_${seed}/"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Slurm variables.

# Sbatch settings
partition="gpu" # l40s, hopper, gpu
qos="normal" # besteffort, iris-hopper, normal
mem="12G"
time="00:00:10"
output="./scripts_sh_V2/output/${model_name}/out_%x_%j.txt"
error="./scripts_sh_V2/output/${model_name}/err_%x_%j.txt"

# Script that will be launched with sbatch. Selected based on the model name.
script_name="./scripts_sh_V2/${model_name}_cent.sh"

# Job name (modify as needed)
job_name="grokking_${model_name}_${seed}"

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

path_dataset_config="./scripts_sh_V2/config/dataset.toml"
path_model_training_config="./scripts_sh_V2/config/training.toml"
path_optimizer_config="./scripts_sh_V2/config/optimizer_${optimizer}.toml"
path_lr_scheduler_config="./scripts_sh_V2/config/lr_sched_${lr_scheduler}.toml"
path_model_config="./scripts_sh_V2/config/model_${model_name}_tiny.toml"
path_model_config="./scripts_sh_V2/config/model_${model_name}.toml"

new_path_dataset_config="./scripts_sh_V2/config/training_hpc/${job_name}/dataset.toml"
new_path_training_config="./scripts_sh_V2/config/training_hpc/${job_name}/training.toml"
new_path_optimizer_config="./scripts_sh_V2/config/training_hpc/${job_name}/optimizer.toml"
new_path_lr_scheduler_config="./scripts_sh_V2/config/training_hpc/${job_name}/lr_scheduler".toml
new_path_model_config="./scripts_sh_V2/config/training_hpc/${job_name}/model.toml"

mkdir -p ./scripts_sh_V2/config/training_hpc/${job_name}/

cp ${path_dataset_config} ${new_path_dataset_config}
cp ${path_model_training_config} ${new_path_training_config}
cp ${path_optimizer_config} ${new_path_optimizer_config}
cp ${path_lr_scheduler_config} ${new_path_lr_scheduler_config}
cp ${path_model_config} ${new_path_model_config}

echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "seed: ${seed}"
echo "path_dataset_config : ${new_path_dataset_config}"
echo "path_to_idx_files   : ${path_to_idx_files}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submit job

sbatch  --job-name=${job_name}\
		--partition=${partition}\
		--qos=${qos}\
		--mem=${mem}\
		--time=${time}\
		--output=${output}\
		--error=${error}\
		${script_name}\
			${path_src}\
			${new_path_dataset_config}\
			${new_path_model_config}\
			${new_path_training_config}\
			${new_path_optimizer_config}\
			${new_path_lr_scheduler_config}\
			${path_to_idx_files}\
			${seed}\
