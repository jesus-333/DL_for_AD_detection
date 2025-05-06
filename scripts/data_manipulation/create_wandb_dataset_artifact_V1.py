"""
Create and log an artifact that contain a dataset.
This is a temporary solution. The dataset folders structure of ADNI require a "refactoring".

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import wandb


from src.dataset import support_dataset_ADNI

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_name = 'ADNI_axial_PD_z_44_slice_4_png_V4_2'
path_to_data = f"./data/{dataset_name}/"

wandb_project_name = 'demnet_training_ADNI'


print_var = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

list_of_path_to_data = [path_to_data + 'AD/', path_to_data + 'CN/', path_to_data + 'MCI/']
file_path_list, label_list_int, label_list_str = support_dataset_ADNI.get_dataset(list_of_path_to_data, print_var = print_var)

if len(list_of_path_to_data) == 0 :
    raise ValueError(f"The list of files is empty. Check the path. Current path = {path_to_data}")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create and log artifact

# Start run
# run = wandb.init(project = wandb_project_name, job_type = "upload-dataset", name = f'upload_{dataset_name}')

# Create artifact
artifact = wandb.Artifact(name = "dataset_name", type = "dataset")

# Add files
n_to_print = len(file_path_list) // 10
for i in range(len(file_path_list)) :
    if (i + 1) % n_to_print == 0 : print(f"Complete {round((i + 1) / len(file_path_list) * 100)}% of loading ({i}/{len(file_path_list)})")
    
    try :
        file_path = file_path_list[i]
        artifact.add_file(local_path = file_path)
    except :
        print(f"Error with file {file_path}. It will be skipped")

# Log files
print("Log artifact")
# run.log_artifact(artifact)
# run.finish()
