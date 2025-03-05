"""
Script to explore and play a little with ADNI 3Yr 3T data.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import nibabel as nib

from src import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_CN  = './data/ADNI1_3Yr_3T/ADNI1_3Yr_3T_CN'
path_AD  = './data/ADNI1_3Yr_3T/ADNI1_3Yr_3T_AD'
path_MCI = './data/ADNI1_3Yr_3T/ADNI1_3Yr_3T_MCI'

file_path_list_CN = support_dataset.get_all_files_from_path(path_CN)
file_path_list_AD = support_dataset.get_all_files_from_path(path_AD)
file_path_list_MCI = support_dataset.get_all_files_from_path(path_MCI)

subj_list_CN = []
for file_path in file_path_list_CN :
    tmp_file_path = file_path.lower()
    subj_id = tmp_file_path.split('/')[3]


data = nib.load(tmp_file_path)
# support_dataset.convert_nii_to_image(data.get_fdata(), './TMP_0/', 0)
# support_dataset.convert_nii_to_image(data.get_fdata(), './TMP_1/', 1)
# support_dataset.convert_nii_to_image(data.get_fdata(), './TMP_2/', 2)
# file_path_list = support_dataset.get_all_files_from_path(path_to_explore)
# data = nib.load(file_path_list[0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# gradwarp_list = []
# b1_list = []
# n3_list = []
#
# all_preprocess_list = []
#
# for file_path in file_path_list :
#     tmp_file_path = file_path.lower()
#
#     if 'gradwarp' in tmp_file_path : 
#         gradwarp_list.append(tmp_file_path)
#         gradwarp = True
#     else : 
#         gradwarp = False
#
#     if 'b1' in tmp_file_path : 
#         b1_list.append(tmp_file_path)
#         b1 = True
#     else : 
#         b1 = False
#     
#     if 'n3' in tmp_file_path : 
#         n3_list.append(tmp_file_path)
#         n3 = True
#     else : 
#         n3 = False
#
#     if gradwarp and b1 and n3 : 
#         all_preprocess_list.append(tmp_file_path)
#
#
# print("Total number of files : {}".format(len(file_path_list)))
# print("Total number of gradwarp files : {}".format(len(gradwarp_list)))
# print("Total number of b1 files : {}".format(len(b1_list)))
# print("Total number of n3 files : {}".format(len(n3_list)))
# print("Total number of files with all preprocess : {}".format(len(all_preprocess_list))) 
