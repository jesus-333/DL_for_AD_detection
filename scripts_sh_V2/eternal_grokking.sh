#!/bin/bash

# Single entry point for the "eternal" grokking runs, i.e. trainings that run for a very large number of epochs on the HPC, using GPUs not used by other people.
# The job body is in eternal_grokking_job.sh. See the header of that file for how the job is kept alive (requeue on preemption and before the time limit).
#
# Usage (from any folder, the script moves to the root of the repository) :
#   bash scripts_sh_V2/eternal_grokking.sh new    <model> [n_runs]          Create n_runs (default 1) new runs (new seeds) and submit them
#   bash scripts_sh_V2/eternal_grokking.sh resume <model> [n_jobs]          Submit up to n_jobs (default 1) runs of the seed list that are not in the queue, not stopped and not finished
#   bash scripts_sh_V2/eternal_grokking.sh import <model> <old_config_folder> <start_epoch>
#                                                                            Create a run that continues an old (not eternal) run, e.g. scripts_sh_V2/config/training_hpc/grokking_vit_371885181/
#                                                                            start_epoch is the number of epochs already done by the old run (the old backups do not store it)
#   bash scripts_sh_V2/eternal_grokking.sh stop   <model> <seed>            Stop a run cleanly at the end of the current epoch (create the STOP file)
#   bash scripts_sh_V2/eternal_grokking.sh status <model>                   Print the state of all the runs in the seed list
#
# <model> is demnet or vit.
#
# Files of a run (job_name = eternal_<model>_<seed>) :
#   scripts_sh_V2/config/training_hpc/<job_name>/   frozen configs, frozen code (code/), idx files (idx/), STOP file    (home, small)
#   model_weights/grokking/<job_name>/               backups A/B (used to resume) and model_BEST.pth                    (home, small)
#   ${SCRATCH}/grokking/<job_name>/                  snapshots every snapshot_every epochs and local wandb files        (scratch)
# The list of all the runs of a model is in scripts_sh_V2/config/seed_list_grokking/<model>.txt (one seed per line).

set -e

# Move to the root of the repository (all the paths below are relative to it)
cd "$(dirname "$0")/.."

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings of NEW runs (modify as needed).
# Note that the settings are frozen when the run is created. Modifying them does not change the runs already created.

# Training
epochs=1000000              # Basically infinite. Use the STOP file to stop a run.
snapshot_every=500          # Save model and optimizer every n epochs (on ${SCRATCH})
batch_size=128
device="cuda"

# Optimizer (AdamW). Constant learning rate (no lr scheduler).
lr=1e-3
betas="0.9 0.99"
weight_decay=1e-2

# Data
path_data="data/ADNI_MRI_Normalized_middle_slice/"
name_tensor_file="dataset_tensor___176_resize.pt"
percentage_train=0.5         # The rest is used for validation (subject-aware split, i.e. subjects in validation are never seen during training)
percentage_validation=0.5
merge_AD_class=0

# Wandb
project_name="ViT_Grokking_Experiment"

# Slurm
partition="gpu"             # l40s, hopper, gpu
qos="besteffort"            # besteffort, iris-hopper, normal
mem="16G"
time="48:00:00"             # Time of a single execution. The job requeues itself 10 minutes before the end.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings that depend on the model

action=$1
model_name=$2

if [ "${model_name}" = "demnet" ] ; then
	template_model_config="./scripts_sh_V2/config/model_demnet.toml"
	python_script="scripts_python/training/demnet_centralized_V2.py"
	conda_env="jesus_hpc_2"
	model_flags="--no-vit_training"
elif [ "${model_name}" = "vit" ] ; then
	template_model_config="./scripts_sh_V2/config/model_vit_tiny.toml"
	python_script="scripts_python/training/vit_transformer_centralized_V2.py"
	conda_env="jesus-hpc"
	model_flags="--vit_training"
else
	echo "Invalid model name: '${model_name}'. Possible values: demnet, vit"
	exit 1
fi

if [ -z "${SCRATCH}" ] ; then
	echo "The variable SCRATCH is not set. It is needed to save snapshots and wandb local files."
	exit 1
fi

path_seed_list="./scripts_sh_V2/config/seed_list_grokking/${model_name}.txt"
path_output="./scripts_sh_V2/output/grokking/"
mkdir -p "${path_output}"
touch "${path_seed_list}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Support functions

# Paths of a run, given the seed
set_run_paths() {
	seed=$1
	job_name="eternal_${model_name}_${seed}"
	path_run_config="./scripts_sh_V2/config/training_hpc/${job_name}/"
	path_run_weights="./model_weights/grokking/${job_name}/"
	path_run_scratch="${SCRATCH%/}/grokking/${job_name}/"
}

# Return 0 if a job with the given name is in the queue (running, pending or requeued)
is_in_queue() {
	[ -n "$(squeue -u "${USER}" -h -n "$1" -o %i)" ]
}

# Random positive seed (31 bits, + 1 to avoid seed = 0)
new_seed() {
	echo $(( ( $(od -An -N4 -tu4 /dev/urandom) & 0x7FFFFFFF ) + 1 ))
}

# Copy the code used by the run (only python files), so the run is not affected by future modifications of the repository.
freeze_code() {
	mkdir -p "${path_run_config}code/"
	rsync -a --prune-empty-dirs --include='*/' --include='*.py' --exclude='*' ./src ./scripts_python "${path_run_config}code/"

	# Save the git state, to know exactly which code was used
	{
		echo "Date   : $(date)"
		echo "Branch : $(git rev-parse --abbrev-ref HEAD)"
		echo "Commit : $(git rev-parse HEAD)"
		echo "Uncommitted changes :"
		git status --short
		git diff
	} > "${path_run_config}code/GIT_INFO.txt"
}

# Update the training config of a run with the settings for the eternal training. Extra arguments are passed to update_training_config.py
update_training_config() {
	python ./scripts_python/training/update_training_config.py\
		--path_training_config="${path_run_config}training.toml"\
		--path_optimizer_config="${path_run_config}optimizer.toml"\
		--path_lr_scheduler_config="${path_run_config}lr_scheduler.toml"\
		--path_to_save_model="${path_run_weights}"\
		--seed=${seed}\
		--epochs=${epochs}\
		--batch_size=${batch_size}\
		--device=${device}\
		--no-use_scheduler\
		--backup_model_every_epoch\
		--resume_from_backup\
		--epoch_to_save_model=${snapshot_every}\
		--path_to_save_snapshot="${path_run_scratch}snapshots/"\
		--stop_file_path="${path_run_config}STOP"\
		--wandb_training\
		--project_name="${project_name}"\
		--name_training_run="${job_name}"\
		--wandb_watch_log=none\
		--no-log_model_artifact\
		--no-fl_training\
		--no-vgg_training\
		--no-swin_training\
		--no-vav_training\
		${model_flags}\
		"$@"
}

submit_run() {
	set_run_paths $1

	if is_in_queue "${job_name}" ; then
		echo "Run ${job_name} is already in the queue. Not submitted."
		return
	fi

	mkdir -p "${path_run_weights}" "${path_run_scratch}snapshots/"

	sbatch  --job-name=${job_name}\
			--partition=${partition}\
			--qos=${qos}\
			--mem=${mem}\
			--time=${time}\
			--output="${path_output}out_%x_%j.txt"\
			--error="${path_output}err_%x_%j.txt"\
			./scripts_sh_V2/eternal_grokking_job.sh\
				${model_name}\
				"${path_run_config}"\
				"${path_run_scratch}"\
				${conda_env}\
				${python_script}
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Actions

if [ "${action}" = "new" ] ; then
	n_runs=${3:-1}

	for i in $(seq 1 ${n_runs}) ; do
		set_run_paths $(new_seed)
		echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
		echo "New run : ${job_name}"

		mkdir -p "${path_run_config}"

		# Configs
		cp ./scripts_sh_V2/config/training.toml "${path_run_config}training.toml"
		cp ./scripts_sh_V2/config/lr_sched_ExponentialLR.toml "${path_run_config}lr_scheduler.toml" # Not used (no scheduler), but required by update_training_config.py
		cp "${template_model_config}" "${path_run_config}model.toml"

		python ./scripts_python/training/update_optimizer.py\
			--path_optimizer_config="${path_run_config}optimizer.toml"\
			--name=AdamW\
			--lr=${lr}\
			--betas ${betas}\
			--weight_decay=${weight_decay}\

		python ./scripts_python/training/update_dataset_config.py\
			--path_dataset_config="${path_run_config}dataset.toml"\
			--path_data="${path_data}"\
			--name_tensor_file="${name_tensor_file}"\
			--merge_AD_class=${merge_AD_class}\
			--percentage_train=${percentage_train}\
			--percentage_validation=${percentage_validation}\
			--percentage_test=0\
			--use_normalization\
			--load_data_in_memory\
			--no-use_rgb_input\
			--no-apply_rescale\

		update_training_config --start_epoch=0
		freeze_code

		echo ${seed} >> "${path_seed_list}"
		submit_run ${seed}
	done

elif [ "${action}" = "import" ] ; then
	path_old_config=${3%/}/
	start_epoch=$4
	if [ ! -d "${path_old_config}" ] || [ -z "${start_epoch}" ] ; then
		echo "Usage: bash scripts_sh_V2/eternal_grokking.sh import <model> <old_config_folder> <start_epoch>"
		exit 1
	fi

	set_run_paths $(python -c "import toml; print(toml.load('${path_old_config}training.toml')['seed'])")
	echo "Import ${path_old_config} as ${job_name} (start epoch ${start_epoch})"
	if [ -d "${path_run_config}" ] ; then
		echo "Run ${job_name} already exists. Use resume instead."
		exit 1
	fi
	mkdir -p "${path_run_config}"

	# Keep the old configs (model, dataset, optimizer) to have a coherent continuation
	for f in dataset model training optimizer lr_scheduler ; do cp "${path_old_config}${f}.toml" "${path_run_config}${f}.toml" ; done

	# Weights of the old run. The older backup is used, because the newest one could be corrupted (see backup_every_epoch in support_training.py)
	path_old_weights=$(python -c "import toml; print(toml.load('${path_old_config}training.toml')['path_to_save_model'])")
	path_past_weights=$(python ./scripts_python/training/get_oldest_file.py --path="${path_old_weights}" --filter="model_backup")
	path_past_optimizer=$(python ./scripts_python/training/get_oldest_file.py --path="${path_old_weights}" --filter="optimizer_backup")
	if [ "${path_past_weights}" = "-" ] || [ "${path_past_optimizer}" = "-" ] ; then
		echo "No backup found in ${path_old_weights}"
		rm -r "${path_run_config}"
		exit 1
	fi
	echo "path_past_weights   : ${path_past_weights}"
	echo "path_past_optimizer : ${path_past_optimizer}"

	update_training_config --start_epoch=${start_epoch} --path_past_weights="${path_past_weights}" --path_past_optimizer="${path_past_optimizer}"

	# Reuse the idx files of the old run, if they still exist (otherwise the job recreates them with the same seed)
	path_old_data=$(python -c "import toml; print(toml.load('${path_old_config}dataset.toml')['path_data'])")
	if [ -d "${path_old_data}CENT_idx_${seed}" ] ; then
		mkdir -p "${path_run_config}idx/"
		cp "${path_old_data}CENT_idx_${seed}/"*.npy "${path_run_config}idx/"
	fi

	freeze_code

	echo ${seed} >> "${path_seed_list}"
	submit_run ${seed}

elif [ "${action}" = "resume" ] ; then
	n_jobs=${3:-1}
	n_submitted=0

	for seed in $(grep -v '^\s*#' "${path_seed_list}") ; do
		[ ${n_submitted} -ge ${n_jobs} ] && break
		set_run_paths ${seed}

		if is_in_queue "${job_name}" ; then continue ; fi
		if [ -f "${path_run_config}STOP" ] ; then continue ; fi
		if [ -f "${path_run_config}FINISHED" ] ; then continue ; fi
		if [ ! -d "${path_run_config}" ] ; then
			echo "Warning: config folder of ${job_name} not found. Skipped."
			continue
		fi

		echo "Resume ${job_name}"
		submit_run ${seed}
		n_submitted=$((n_submitted + 1))
	done

	echo "${n_submitted} run(s) submitted."

elif [ "${action}" = "stop" ] ; then
	set_run_paths $3
	if [ ! -d "${path_run_config}" ] ; then
		echo "Run ${job_name} not found."
		exit 1
	fi
	touch "${path_run_config}STOP"
	echo "STOP file created for ${job_name}. The training stops at the end of the current epoch (if the job is pending, it exits immediately when it starts)."
	echo "To restart the run, delete ${path_run_config}STOP and use resume."

elif [ "${action}" = "status" ] ; then
	printf "%-30s %-12s %s\n" "RUN" "SLURM" "NOTES"
	for seed in $(grep -v '^\s*#' "${path_seed_list}") ; do
		set_run_paths ${seed}
		state=$(squeue -u "${USER}" -h -n "${job_name}" -o %T | head -n 1)
		notes=""
		[ -f "${path_run_config}STOP" ] && notes="${notes} STOPPED"
		[ -f "${path_run_config}FINISHED" ] && notes="${notes} FINISHED"
		printf "%-30s %-12s %s\n" "${job_name}" "${state:--}" "${notes}"
	done

else
	echo "Invalid action: '${action}'. Possible values: new, resume, import, stop, status"
	echo "See the header of this script for the usage."
	exit 1
fi
