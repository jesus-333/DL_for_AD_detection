"""
Similar to V1. The folder structure of the data is preserved during log

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import wandb
import os

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = 'ADNI_axial_PD_z_44_slice_4_png_V4_2'
path_to_data = f"./data/{dataset_name}/"

wandb_project_name = 'demnet_training_ADNI'


print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

folders_paths_dict = support_dataset.get_all_files_from_path_divided_per_folder(path_to_data, filetype_filter = 'png')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create and log artifact

# Start run
run = wandb.init(project = wandb_project_name, job_type = "upload-dataset", name = f'upload_{dataset_name}')

# Create artifact
artifact = wandb.Artifact(name = f"{dataset_name}", type = "dataset")

# Add files
n_to_print = len(folders_paths_dict) // 10
i = 0
for folder_path in folders_paths_dict :
	if (i + 1) % n_to_print == 0 : print(f"Complete {round((i + 1) / len(folders_paths_dict) * 100)}% of loading ({i}/{len(folders_paths_dict)})")

	folder_name = folder_path.split('/')[-2]

	for file_path in folders_paths_dict[folder_path] :

		file_name = file_path.split('/')[-1]

		try :
			name = os.path.join(folder_name, file_name)
			artifact.add_file(local_path = file_path, name = name)
		except :
			print(f"Error with file {file_path}. It will be skipped")

	i += 1

# Log files
print("Log artifact")
run.log_artifact(artifact)
run.finish()
