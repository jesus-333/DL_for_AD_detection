# Path to library
PATH_SRC="./"

# Paths to config files
PATH_CONFIG_FOLDER="./config/vgg_finetuning/"
PATH_DATASET_CONFIG="${PATH_CONFIG_FOLDER}dataset.toml"
PATH_MODEL_CONFIG="${PATH_CONFIG_FOLDER}model_vgg.toml"
PATH_TRAINING_CONFIG="${PATH_CONFIG_FOLDER}training.toml"
PATH_LR_SCHEDULER_CONFIG="${PATH_CONFIG_FOLDER}lr_scheduler_config.toml"

# Path to data
PATH_DATA="data/ADNI_axial_middle_slice/" 
NAME_TENSOR_FILE="dataset_tensor___176_resize.pt"
# Remember to change apply_rescale to no-apply_rescale if you do not used data saved in interger (like the middle_slice)

# Dataset settings
merge_AD_class=1
percentage_train=0.8
percentage_validation=0.1
percentage_test=0.1
rescale_factor=4095

# Training settings
batch_size=128
lr=1e-4
epochs=60
device="cuda"
epoch_to_save_model=-1
path_to_save_model="model_weights_ADNI"
seed=-1
vgg_training_mode=0

# Wandb Settings
name_training_run="vgg_trainin_mode_${vgg_training_mode}_fixed_lr_epochs_${epochs}_batch_${batch_size}"

# Always check use_vgg_normalization_values and use_rgb_input, use_pretrained_vgg

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

python ./scripts_python/training/reset_config_files.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--path_training_config="${PATH_TRAINING_CONFIG}"\

python ./scripts_python/training/update_dataset_config.py\
	--path_dataset_config="${PATH_DATASET_CONFIG}"\
	--no-filter_AD_data\
	--merge_AD_class=${merge_AD_class}\
	--percentage_train=${percentage_train}\
	--percentage_validation=${percentage_validation}\
	--percentage_test=${percentage_test}\
	--apply_rescale\
	--rescale_factor=${rescale_factor}\
	--use_normalization\
	--load_data_in_memory\
	--no-use_rgb_input\
	
python ./scripts_python/training/update_training_config.py\
	--path_training_config="${PATH_TRAINING_CONFIG}"\
	--path_lr_scheduler_config="${PATH_LR_SCHEDULER_CONFIG}"\
	--batch_size=${batch_size}\
	--lr=${lr}\
	--epochs=${epochs}\
	--device="${device}"\
	--epoch_to_save_model=${epoch_to_save_model}\
	--path_to_save_model="${path_to_save_model}"\
	--seed=${seed}\
	--no-use_scheduler\
	--measure_metrics_during_training\
	--print_var\
	--vgg_training\
	--use_pretrained_vgg\
	--vgg_training_mode=${vgg_training_mode}\
	--no-use_vgg_normalization_values\
	--wandb_training\
	--project_name="vgg_ADNI"\
	--name_training_run="${name_training_run}"\
	--model_artifact_name="${model_artifact_name}"\
	--log_freq=1\
	--no-log_model_artifact
	--no-debug\
