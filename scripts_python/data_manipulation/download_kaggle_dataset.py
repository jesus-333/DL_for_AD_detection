"""
Download the dataset from kaggle and save it in a specific directory
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import kagglehub

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_to_save_data = './data/kaggle_AD_4_classes_automatic_download/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path = kagglehub.dataset_download("marcopinamonti/alzheimer-mri-4-classes-dataset")
