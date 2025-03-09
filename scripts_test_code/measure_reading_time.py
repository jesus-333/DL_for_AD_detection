"""
Measure how long it takes to read a DCM files and a png files.
For the DCM files also the conversion time in tensore is evaluated
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import torchvision
import pydicom as dicom
import time
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_png = './data/ADNI_1_3YR_3T_above_all/AD/002_S_1018/2006-12-18_60.png'
path_dcm = './data/ADNI_3_fMRI/AD/941_S_6962/Axial_MB_rsfMRI__Eyes_Open_/2021-07-13_11_46_55.0/I1467536/ADNI_941_S_6962_MR_Axial_MB_rsfMRI__Eyes_Open___br_raw_20210713134939917_1_S1042915_I1467536.dcm'

n_repetition = 200

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

list_time_png = []
list_time_dcm = []
list_time_dcm_tensor = []

iteration_to_print_status = int(n_repetition / 10)
if iteration_to_print_status <= 0 : iteration_to_print_status = 1

for i in range(n_repetition) :
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Start time for DCM
    start_time = time.time()

    # Read the dcm file
    image = dicom.dcmread(path_dcm)

    # Measure the time (without the conversion to tensor)
    end_time = time.time()
    list_time_png.append(end_time - start_time)

    # Convert to tensor and measure the time
    image_tensor = torch.tensor(image.pixel_array)
    end_time = time.time()
    list_time_dcm_tensor.append(end_time - start_time)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Start time for PNG
    start_time = time.time()

    # Read the png file
    image = torchvision.io.read_image(path_png)

    # Measure the time
    end_time = time.time()
    list_time_dcm.append(end_time - start_time)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    if i % iteration_to_print_status == 0 : print("Repetitions complete = {:.2f}%".format((i) / n_repetition * 100))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Print the results (mean and std)
print('PNG file')
print('{:.3f}±{:.3f}ms'.format(np.mean(list_time_png) * 1000, np.std(list_time_png) * 1000))

print('DCM file')
print('{:.3f}±{:.3f}ms'.format(np.mean(list_time_dcm) * 1000, np.std(list_time_dcm) * 1000))

print('DCM file tensor')
print('{:.3f}±{:.3f}ms'.format(np.mean(list_time_dcm_tensor) * 1000, np.std(list_time_dcm_tensor) * 1000))






