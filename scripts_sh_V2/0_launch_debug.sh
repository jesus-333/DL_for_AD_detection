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
model_name="swin"
optimizer="AdamW"
lr_scheduler="ExponentialLR"

# Possible lr lr_scheduler ExponentialLR, CosineAnnealingWarmRestarts
# Possible optimizer AdamW, SGD

# Data paths and names.
path_data="data/ADNI_axial_middle_slice/" 
name_tensor_file="dataset_tensor___176_resize.pt"
path_to_idx_files="${path_data}CENT_idx_${seed}/"
percentage_data_used_for_training=0.8

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Slurm variables.

# Sbatch settings
partition="gpu"
qos="normal"
mem="16G"
time="01:00:00"
output="./scripts_sh_V2/output/out_%x_%j.txt"
error="./scripts_sh_V2/output/err_%x_%j.txt"

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

mkdir -p ./scripts_sh_V2/config/training_hpc/${job_name}/

cp ${path_dataset_config} ./scripts_sh_V2/config/training_hpc/${job_name}/dataset.toml
cp ${path_model_training_config} ./scripts_sh_V2/config/training_hpc/${job_name}/training.toml
cp ${path_optimizer_config} ./scripts_sh_V2/config/training_hpc/${job_name}/optimizer.toml
cp ${path_lr_scheduler_config} ./scripts_sh_V2/config/training_hpc/${job_name}/lr_scheduler.toml
cp ${path_model_config} ./scripts_sh_V2/config/training_hpc/${job_name}/model.toml

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submit job
sh ${script_name}\
	${path_src}\
	./scripts_sh_V2/config/training_hpc/${job_name}/dataset.toml\
	./scripts_sh_V2/config/training_hpc/${job_name}/model.toml\
	./scripts_sh_V2/config/training_hpc/${job_name}/training.toml\
	./scripts_sh_V2/config/training_hpc/${job_name}/optimizer.toml\
	./scripts_sh_V2/config/training_hpc/${job_name}/lr_scheduler.toml\
	${path_to_idx_files}\
	${seed}\
