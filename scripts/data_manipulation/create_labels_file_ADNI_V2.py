"""
This files work exactly as V1 but it's created to handle the CSV obtained with the simpler image collection creator.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import pandas
import json

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

name_dataset = 'ADNI_axial_PD_z_44_slice_4'
path_labels = f'./data/ADNI_Labels/{name_dataset}.csv'
path_labels = './data/ADNI_Labels/ALL_subject_MRI.csv'

label_to_int = dict(
    CN    = 0,
    AD    = 1,
    MCI   = 2,
    EMCI  = 3,
    LMCI  = 4,
    SMC   = 5,
    Patient = 6,
)

df = pandas.read_csv(path_labels)

dict_to_save_str = dict()
dict_to_save_int = dict()

for i in range(len(df)) :
    row = df.iloc[i]
    subj_id = row['Subject']
    label = row['Group']
    dict_to_save_str[subj_id] = label
    dict_to_save_int[subj_id] = label_to_int[label]

# Save dictionary
with open(f'./data/ADNI_Labels/{name_dataset}_str.json', 'w') as f : json.dump(dict_to_save_str, f)
with open(f'./data/ADNI_Labels/{name_dataset}_int.json', 'w') as f : json.dump(dict_to_save_int, f)
