#!/bin/bash

# Job body of the "eternal" grokking runs. Do NOT launch this script directly, use eternal_grokking.sh (it passes all the arguments and the sbatch settings).
# The job trains the model until training_config['epochs'] is reached or the STOP file is created. Each time the job is (re)started the training is resumed from the last valid checkpoint.
#
# How the job is kept alive :
# - If the job is preempted (e.g. qos besteffort) SLURM requeues it automatically (--requeue).
# - Some minutes before the time limit, SLURM sends the signal USR1 to this script (--signal=B:USR1@...). The trap below requeues the job itself.
#   Without this, the job would end with TIMEOUT and it would not be requeued.
# - If the python script crashes (exit code != 0) the job is NOT requeued, to avoid an infinite loop of crashes. Use `eternal_grokking.sh resume` after fixing the problem.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get arguments passed to the script.

model_name=$1           # demnet or vit
path_run_config=$2      # Folder with the frozen configs, frozen code and idx files of this run
path_run_scratch=$3     # Folder on ${SCRATCH} for snapshots and wandb local files
conda_env=$4            # Name of the conda environment to activate
python_script=$5        # Training script (relative to the frozen code folder)

path_code="${path_run_config}code/"
path_idx="${path_run_config}idx/"
path_stop_file="${path_run_config}STOP"

echo "==================================================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Restart count : ${SLURM_RESTART_COUNT:-0}"
echo "Node          : $(hostname)"
echo "Date          : $(date)"
echo "model_name       : ${model_name}"
echo "path_run_config  : ${path_run_config}"
echo "path_run_scratch : ${path_run_scratch}"
echo "python_script    : ${python_script}"
echo "==================================================="

# If the STOP file exists the run was stopped on purpose. Nothing to do.
if [ -f "${path_stop_file}" ] ; then
	echo "STOP file found (${path_stop_file}). Exit without training."
	exit 0
fi

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load python environment

eval "$(conda shell.bash hook)"
conda activate ${conda_env}
echo "Conda environment : ${CONDA_DEFAULT_ENV}"

# Use the frozen copy of the code instead of the package installed in the environment.
# PYTHONPATH has priority over the site-packages, so `import addl` loads the frozen code.
# No `pip install .` here : the code must not change during the run and several jobs could install the package at the same time in the same environment.
export PYTHONPATH="${path_code}src:${PYTHONPATH}"
python -c "import addl, os; print('addl imported from', os.path.dirname(addl.__file__))"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Wandb settings (avoid filling the disk quota)

# All the local wandb files go on ${SCRATCH}, in a folder used only by this run.
# The folder is deleted each time the job starts. At this point no process is writing in it, and the data of the previous executions are already synced on the wandb server.
export WANDB_DIR="${path_run_scratch}wandb_local/"
export WANDB_CACHE_DIR="${path_run_scratch}wandb_cache/"
export WANDB_DATA_DIR="${path_run_scratch}wandb_data/"
export WANDB_ARTIFACT_DIR="${path_run_scratch}wandb_artifacts/"
rm -rf "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}" "${WANDB_ARTIFACT_DIR}"

# Do not save a copy of the stdout in the wandb folder (the stdout is already saved by SLURM).
export WANDB_CONSOLE=off

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Prepare data (only the first time).
# The split is subject-aware and deterministic given the seed. It is created only once, so all the executions of this run use exactly the same files.

if [ ! -f "${path_idx}train_idx_all.npy" ] || [ ! -f "${path_idx}val_idx.npy" ] ; then
	echo "idx files not found. Create them."

	path_data=$(python -c "import toml; print(toml.load('${path_run_config}dataset.toml')['path_data'])")
	name_tensor_file=$(python -c "import toml; print(toml.load('${path_run_config}dataset.toml')['name_tensor_file'])")
	percentage_train=$(python -c "import toml; print(toml.load('${path_run_config}dataset.toml')['percentage_train'])")
	seed=$(python -c "import toml; print(toml.load('${path_run_config}training.toml')['seed'])")

	python "${path_code}scripts_python/data_manipulation/create_idx_files_for_federated_simulations_2.py"\
		--path_data="${path_data}"\
		--name_tensor_file="${name_tensor_file}"\
		--path_to_save="${path_idx}"\
		--percentage_data_used_for_training=${percentage_train}\
		--num_clients=3\
		--seed=${seed}\
		--no-use_cross_fold_validation\
		--keep_samples_proportion\

	echo "Data preparation END"
fi

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Requeue before the time limit

requeued=0
requeue_job() {
	echo "Signal USR1 received at $(date) : time limit is close. Requeue job ${SLURM_JOB_ID}."
	requeued=1
	scontrol requeue ${SLURM_JOB_ID}
}
trap requeue_job USR1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train
# The training runs in background, otherwise bash executes the trap only after the end of the python process (i.e. too late).

python "${path_code}${python_script}"\
	--path_src="${path_code}"\
	--path_dataset_config="${path_run_config}dataset.toml"\
	--path_model_config="${path_run_config}model.toml"\
	--path_training_config="${path_run_config}training.toml"\
	--path_to_idx_files="${path_idx}" &

pid_training=$!
wait ${pid_training}
exit_code=$?

if [ ${requeued} -eq 1 ] ; then
	# The job was requeued by the trap. SLURM will kill this execution, wait for it.
	wait ${pid_training}
	exit 0
fi

echo "Training ended with exit code ${exit_code} at $(date)"

# Mark the run as finished only if all the epochs were done (i.e. the training was not stopped with the STOP file).
# Note that model_END.pth is created also when the training is stopped with the STOP file, so it cannot be used for this.
if [ ${exit_code} -eq 0 ] && [ ! -f "${path_stop_file}" ] ; then
	touch "${path_run_config}FINISHED"
fi

exit ${exit_code}
