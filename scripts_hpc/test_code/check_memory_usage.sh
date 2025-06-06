#!/bin/bash -l

#SBATCH --job-name="check_memory_usage_hpc"
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=0-00:05:00
#SBATCH --qos=normal
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./scripts_hpc/output/std_output_%x_%j.txt
#SBATCH --error=./scripts_hpc/output/other_output_%x_%j.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

conda activate jesus-hpc

#conda list
#pip list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Path to data
PATH_DATA="data/ADNI_axial_3D_z_16_size_176_int/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize___int.pt"

n_repetitions=5
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

srun python ./scripts_python/test_code/check_memory_usage_load_data.py \
	--path_src="${PATH_SRC}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"\
	--n_repetitions=${n_repetitions}
	
