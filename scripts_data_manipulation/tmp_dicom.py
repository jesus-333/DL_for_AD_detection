import pydicom as dicom

from src.dataset import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_AD = 'data/ADNI_3_fMRI/AD'
path_MCI = 'data/ADNI_3_fMRI/MCI'
path_CN = 'data/ADNI_3_fMRI/CN'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

file_path_list_CN = support_dataset.get_all_files_from_path(path_CN)
file_path_list_AD = support_dataset.get_all_files_from_path(path_AD)
file_path_list_MCI = support_dataset.get_all_files_from_path(path_MCI)
