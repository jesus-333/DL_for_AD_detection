#!/bin/sh

# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="./scripts_hpc/config_demnet_wandb/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"

# Path to data
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"

# (DEBUG) Check path
echo "${PATH_SRC}"
echo "${PATH_DATASET_CONFIG}"
echo "${PATH_MODEL_CONFIG}"
echo "${PATH_TRAINING_CONFIG}"

python ./scripts/training/demnet_ADNI_wandb_V3.py \
	--path_src="${PATH_SRC}"\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_model_config="${PATH_MODEL_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_data="${PATH_DATA}"\
	--name_tensor_file="${NAME_TENSOR_FILE}"\

# parser.add_argument('-p_data'   , '--path_data'             , type = str, default = None)
# parser.add_argument('-n'        , '--name_tensor_file'      , type = str, default = None)
