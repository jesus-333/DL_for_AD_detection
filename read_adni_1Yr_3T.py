"""
Script to explore and play a little with ADNI 1Yr 3T data.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import nibabel as nib

from src import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


path_to_explore = './data/ADNI_1Yr_3T_All_mixed'
file_path_list = support_dataset.get_all_files_from_path(path_to_explore)

data = nib.load(file_path_list[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

gradwarp_list = []
b1_list = []
n3_list = []

all_preprocess_list = []

for file_path in file_path_list :
    tmp_file_path = file_path.lower()

    if 'gradwarp' in tmp_file_path : 
        gradwarp_list.append(tmp_file_path)
        gradwarp = True
    else : 
        gradwarp = False

    if 'b1' in tmp_file_path : 
        b1_list.append(tmp_file_path)
        b1 = True
    else : 
        b1 = False
    
    if 'n3' in tmp_file_path : 
        n3_list.append(tmp_file_path)
        n3 = True
    else : 
        n3 = False

    if gradwarp and b1 and n3 : 
        all_preprocess_list.append(tmp_file_path)


print("Total number of files : {}".format(len(file_path_list)))
print("Total number of gradwarp files : {}".format(len(gradwarp_list)))
print("Total number of b1 files : {}".format(len(b1_list)))
print("Total number of n3 files : {}".format(len(n3_list)))
print("Total number of files with all preprocess : {}".format(len(all_preprocess_list))) 
