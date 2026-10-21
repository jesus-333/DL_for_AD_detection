#!/bin/sh

#SBATCH --job-name="train_demnet_CENTRALIZED_subject_aware_split"
#SBATCH --nodes=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=13G
#SBATCH --time=0-02:00:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_sh/train_scripts_submission/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_sh/train_scripts_submission/output/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Centralized counterpart of demnet_FL_subject_aware_partitioning.sh
# Data split: subject-aware split (all the samples of a subject are either in training or in validation)
# This training used
# - AdamW optimizer
# - ExponentialLR
#
# The script has two modes, selected by the variable run_tag:
# - run_tag < 0  : NEW settings. All the configs are created from scratch with the values defined in this script.
# - run_tag = 0  : REUSE the configs and the data split of previous FL training runs (to compare centralized vs FL with the exact same settings).
#                  The tags are read from the file old_run_tags.txt (same folder of this script), one tag per row. One repetition for each row (n_repetitions is ignored).
# - run_tag > 0  : REUSE the configs and the data split of a single FL training run, i.e. run_tag is the tag of that run (single repetition).
# The tag of a FL run is "<slurm job id>_<repetition>" (e.g. 5254840_2), see the variable run_tag inside demnet_FL_subject_aware_partitioning.sh
# In reuse mode, the configs (dataset, model, training) and the files train_idx_all.npy and val_idx.npy of the FL run are COPIED, so the FL files are never modified.
# Only the settings that depend on the training method (centralized vs FL) are changed: epochs, device, wandb, path to save the model and fl_training.
#
# WARNING : in wandb the id of a run is its seed (see wandb_train in train_functions.py) and the run is resumed if the id already exists in the project.
# In reuse mode the seed is the one of the FL run. This is not a problem as long as the centralized and the FL runs are logged in different wandb projects (as it is now).
# But if you launch twice the same tag in the same wandb project (or if you log centralized and FL runs in the same project) the runs will overwrite each other.
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# set -euo pipefail

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

# Tag of the FL run(s) to reuse (see the header of the script). Negative = new settings.
run_tag=-1
PATH_FILE_RUN_TAGS="./scripts_sh/train_scripts_submission/old_run_tags.txt"

# Path to library
PATH_SRC="./"

# Paths to config files of the FL training (used only in reuse mode, they are never modified)
PATH_CONFIG_FOLDER_FL="scripts_python/training_FL/ADNI_demnet_fedavg_with_wandb_V2/config/"
PATH_MODEL_CONFIG_TEMPLATE="${PATH_CONFIG_FOLDER_FL}template/model.toml"

# Paths to config files of the centralized training
PATH_CONFIG_FOLDER="scripts_python/training/config_centralized_paper/"
mkdir -p ${PATH_CONFIG_FOLDER}

# Path to data
PATH_DATA="data/ADNI_MRI_Normalized_middle_slice/"  # Note that the normalized dataset has only 3 classes : AD, CN and MCI
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"
# N.B. The file should be already saved with value alreay normalized between 0 and 1. 

# Data preparation settings (used only if run_tag < 0)
percentage_data_used_for_training=0.9
n_repetitions=5

# Dataset settings (used only if run_tag < 0)
merge_AD_class=0
percentage_train=0.85
percentage_validation=0.15
percentage_test=0
rescale_factor=4095

# Training settings
# Note that in reuse mode only epochs, device, epoch_to_save_model and path_to_save_model are used
batch_size=192
epochs=100
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights/demnet_ADNI_CENTRALIZED/subject_aware_split"

# Optimizer config (used only if run_tag < 0)
lr=1e-3
name_optimizer='AdamW'
beta_low=0.9
beta_high=0.999
eps=1e-8
weight_decay=1e-5

# Lr scheduler settings (used only if run_tag < 0)
gamma=0.95

# Information about data used for model_config (used only if run_tag < 0)
input_channels=1
input_size=176

# Wandb Setting
project_name="demnet_centralized_subject_aware_split_PAPER"
name_training_run="centralized"
model_artifact_name="demnet_z_${input_channels}"

# Always check use_rgb_input
# Remember also to check the wandb settings (e.g. the log_model_artifact parameter). Currently no artifact is logged.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Select the mode and (in reuse mode) load the tags

case "${run_tag}" in
	-*)
		reuse_config=0
		;;
	0)
		reuse_config=1
		if [ ! -f "${PATH_FILE_RUN_TAGS}" ] ; then echo "ERROR: file ${PATH_FILE_RUN_TAGS} not found"; exit 1; fi
		# Empty rows are ignored. The number of repetitions is the number of rows
		list_run_tags=$(grep -v '^[[:space:]]*$' "${PATH_FILE_RUN_TAGS}" | tr -d '[:blank:]')
		n_repetitions=$(echo "${list_run_tags}" | grep -c .)
		if [ $n_repetitions -eq 0 ] ; then echo "ERROR: file ${PATH_FILE_RUN_TAGS} is empty"; exit 1; fi
		;;
	*)
		reuse_config=1
		list_run_tags="${run_tag}"
		n_repetitions=1
		;;
esac
echo "reuse_config = ${reuse_config} - n_repetitions = ${n_repetitions}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Repeate the experiment n_repetitions times
for repetition in $(seq 1 $n_repetitions); do
	echo "====================================="
	echo "REPETITION ${repetition} / ${n_repetitions}"
	echo "====================================="

	# Tag of this centralized run. All the files created in this repetition contain it, so a repetition never overwrites another one.
	current_tag="${SLURM_JOB_ID}_${repetition}"

	PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset_${current_tag}.toml"
	PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model_${current_tag}.toml"
	PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training_${current_tag}.toml"
	PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER}optimizer_config_${current_tag}.toml"
	PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config_${current_tag}.toml"
	path_to_save_idx_file="${PATH_DATA}CENT_idx_${current_tag}/"
	path_to_save_model_run="${path_to_save_model}_${current_tag}"

	if [ $reuse_config -eq 0 ] ; then
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# NEW settings

		# Seed unique for each repetition (hash of the tag, same procedure of the FL scripts)
		seed=$(python -c "import hashlib; print(int(hashlib.sha256('${current_tag}'.encode()).hexdigest(), 16) % (2**31 - 1) + 1)")
		echo "RUN TAG ${current_tag} - SEED ${seed}"

		# Prepare data
		# The script creates the idx files used for FL simulations. Here we use a single "client" so that train_idx_all.npy contains all the training data.
		srun python ./scripts_python/data_manipulation/create_idx_files_for_federated_simulations_2.py\
			--path_data=${PATH_DATA}\
			--name_tensor_file=${NAME_TENSOR_FILE}\
			--path_to_save=${path_to_save_idx_file}\
			--percentage_data_used_for_training=${percentage_data_used_for_training}\
			--num_clients=1\
			--seed=${seed}\
			--no-use_cross_fold_validation\
			--keep_samples_proportion\

		# Reset config files
		srun python ./scripts_python/training/reset_config_files.py\
			--path_dataset_config="${PATH_DATASET_CONFIG}"\
			--path_training_config="${PATH_TRAINING_CONFIG}"\

		# Update model config
		num_classes=3
		echo "NUM CLASSES ${num_classes}"

		srun python ./scripts_python/training/update_model_config_demnet.py\
			--path_save=${PATH_MODEL_CONFIG}\
			--path_template=${PATH_MODEL_CONFIG_TEMPLATE}\
			--input_channels=${input_channels}\
			--input_size=${input_size}\
			--num_classes=${num_classes}

		# Update dataset config
		srun python ./scripts_python/training/update_dataset_config.py\
			--path_dataset_config="${PATH_DATASET_CONFIG}"\
			--path_data=${PATH_DATA}\
			--name_tensor_file=${NAME_TENSOR_FILE}\
			--path_idx_folder=${path_to_save_idx_file}\
			--merge_AD_class=${merge_AD_class}\
			--percentage_train=${percentage_train}\
			--percentage_validation=${percentage_validation}\
			--percentage_test=${percentage_test}\
			--no-apply_rescale\
			--rescale_factor=${rescale_factor}\
			--use_normalization\
			--no-use_rgb_input\
			--load_data_in_memory\

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
			--name="ExponentialLR"\
			--gamma=${gamma}\

		# Update training config
		srun python ./scripts_python/training/update_training_config.py\
			--path_training_config="${PATH_TRAINING_CONFIG}"\
			--path_optimizer_config="${PATH_OPTIMIZER_CONFIG}"\
			--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
			--batch_size=${batch_size}\
			--epochs=${epochs}\
			--device="${device}"\
			--epoch_to_save_model=${epoch_to_save_model}\
			--path_to_save_model="${path_to_save_model_run}"\
			--seed=${seed}\
			--use_scheduler\
			--measure_metrics_during_training\
			--no-fl_training\
			--print_var\
			--wandb_training\
			--project_name=${project_name}\
			--entity="alberto_zancanaro_academic"\
			--model_artifact_name="${model_artifact_name}"\
			--name_training_run="${name_training_run}"\
			--no-log_model_artifact\
			--log_freq=1\

	else
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# REUSE the settings of a FL training run

		old_tag=$(echo "${list_run_tags}" | sed -n "${repetition}p")
		echo "REUSE FL RUN ${old_tag} - CENTRALIZED RUN ${current_tag}"

		# Paths of the FL run (never modified)
		OLD_PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER_FL}dataset_${old_tag}.toml"
		OLD_PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER_FL}model_${old_tag}.toml"
		OLD_PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER_FL}training_${old_tag}.toml"
		OLD_PATH_OPTIMIZER_CONFIG="${PATH_CONFIG_FOLDER_FL}optimizer_config_${old_tag}.toml"
		OLD_PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER_FL}lr_scheduler_config_${old_tag}.toml"
		OLD_PATH_IDX_FILE="${PATH_DATA}FL_idx_${old_tag}/"

		# Check that everything exists
		for f in ${OLD_PATH_DATASET_CONFIG} ${OLD_PATH_MODEL_CONFIG} ${OLD_PATH_TRAINING_CONFIG} ${OLD_PATH_OPTIMIZER_CONFIG} ${OLD_PATH_LR_SCHEDULER_CONFIG} ${OLD_PATH_IDX_FILE}train_idx_all.npy ${OLD_PATH_IDX_FILE}val_idx.npy ; do
			if [ ! -f "$f" ] ; then echo "ERROR: file $f not found"; exit 1; fi
		done

		# Seed of the FL run
		seed=$(grep -m1 '^seed' ${OLD_PATH_TRAINING_CONFIG} | sed 's/[^0-9]//g')
		if [ -z "${seed}" ] ; then echo "ERROR: seed not found in ${OLD_PATH_TRAINING_CONFIG}"; exit 1; fi
		echo "SEED ${seed} (from the FL run)"

		# Copy data split (only the files needed by the centralized training) and configs
		mkdir -p ${path_to_save_idx_file}
		cp ${OLD_PATH_IDX_FILE}train_idx_all.npy ${OLD_PATH_IDX_FILE}val_idx.npy ${path_to_save_idx_file}
		cp ${OLD_PATH_DATASET_CONFIG} ${PATH_DATASET_CONFIG}
		cp ${OLD_PATH_MODEL_CONFIG} ${PATH_MODEL_CONFIG}

		# Create the training config from the one of the FL run, changing only what depends on the training method.
		# Note that update_training_config.py re-sets (and therefore these arguments must be passed explicitly): seed, device, epoch_to_save_model, path_to_save_model, optimizer config, lr scheduler config and the boolean flags
		# The arguments not passed (e.g. batch_size) are kept as in the FL training config.
		srun python ./scripts_python/training/update_training_config.py\
			--path_training_config="${OLD_PATH_TRAINING_CONFIG}"\
			--path_to_save="${PATH_TRAINING_CONFIG}"\
			--path_optimizer_config="${OLD_PATH_OPTIMIZER_CONFIG}"\
			--path_lr_scheduler_config="${OLD_PATH_LR_SCHEDULER_CONFIG}"\
			--epochs=${epochs}\
			--device="${device}"\
			--epoch_to_save_model=${epoch_to_save_model}\
			--path_to_save_model="${path_to_save_model_run}"\
			--seed=${seed}\
			--use_scheduler\
			--measure_metrics_during_training\
			--no-fl_training\
			--print_var\
			--wandb_training\
			--project_name=${project_name}\
			--entity="alberto_zancanaro_academic"\
			--model_artifact_name="${model_artifact_name}"\
			--name_training_run="${name_training_run}"\
			--no-log_model_artifact\
			--log_freq=1\

	fi

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Launch CENTRALIZED Training

	srun python ./scripts_python/training/demnet_centralized_V2.py\
		--path_src="${PATH_SRC}"\
		--path_dataset_config="${PATH_DATASET_CONFIG}"\
		--path_model_config="${PATH_MODEL_CONFIG}"\
		--path_training_config="${PATH_TRAINING_CONFIG}"\
		--path_to_idx_files=${path_to_save_idx_file}\

	echo "END Repetition"

done # End of the for loop for repetitions
