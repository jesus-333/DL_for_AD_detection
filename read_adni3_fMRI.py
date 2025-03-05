
import pydicom as dicom
import matplotlib.pylab as plt
import cv2
import os

from src import support_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_save = 'TMP_fMRI/'

file_path_list = support_dataset.get_all_files_from_path('./data/ADNI/')

os.makedirs(path_save, exist_ok = True)

for i in range(len(file_path_list)) :
    try :
        ds = dicom.dcmread(file_path_list[i])

        cv2.imwrite('{}{}.png'.format(path_save, i), ds.pixel_array) 
        print(i)
    except :
        print(f"Error with path number {i}")
