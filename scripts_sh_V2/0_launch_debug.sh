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
model_name="vav"
optimizer="AdamW"
lr_scheduler="ExponentialLR"

# Remember that for demnet you have to set use_rgb_input to false inside dataset.toml
# Also if you have problem with NUM_CLASSES and MERGE_AD_CLASS remember to read the note you write in demnet.sh, at the end of `Prepare data` section.

# Possible optimizer AdamW, SGD
# Possible lr lr_scheduler ExponentialLR, CosineAnnealingWarmRestarts
# IF you do not want to use any LR scheduler you have to add the optional flag --no-use_scheduler\ directly inside the script.

# Data paths and names.
# Remember that to modify the dataset you have ALSO to modify the data path inside the dataset.toml
path_data="data/3D_MRI_data_for_test/" 
path_to_idx_files="${path_data}CENT_idx_${seed}/"
percentage_data_used_for_training=0.9

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Script variables (equivalent to the slurm variables section of the 0_launch_sbatch_cent).

# Script that will be launched with sbatch. Selected based on the model name.
script_name="./scripts_sh_V2/${model_name}_cent.sh"

# Job name (modify as needed)
job_name="train_${model_name}_${seed}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create config file for training run.

path_dataset_config="./scripts_sh_V2/config/dataset.toml"
path_model_training_config="./scripts_sh_V2/config/training.toml"
path_optimizer_config="./scripts_sh_V2/config/optimizer_${optimizer}.toml"
path_lr_scheduler_config="./scripts_sh_V2/config/lr_sched_${lr_scheduler}.toml"
path_model_config="./scripts_sh_V2/config/model_${model_name}.toml"

mkdir -p ./scripts_sh_V2/config/debug/${job_name}/

cp ${path_dataset_config} ./scripts_sh_V2/config/debug/${job_name}/dataset.toml
cp ${path_model_training_config} ./scripts_sh_V2/config/debug/${job_name}/training.toml
cp ${path_optimizer_config} ./scripts_sh_V2/config/debug/${job_name}/optimizer.toml
cp ${path_lr_scheduler_config} ./scripts_sh_V2/config/debug/${job_name}/lr_scheduler.toml
cp ${path_model_config} ./scripts_sh_V2/config/debug/${job_name}/model.toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submit job
sh ${script_name}\
	${path_src}\
	./scripts_sh_V2/config/debug/${job_name}/dataset.toml\
	./scripts_sh_V2/config/debug/${job_name}/model.toml\
	./scripts_sh_V2/config/debug/${job_name}/training.toml\
	./scripts_sh_V2/config/debug/${job_name}/optimizer.toml\
	./scripts_sh_V2/config/debug/${job_name}/lr_scheduler.toml\
	${path_to_idx_files}\
	${seed}\
