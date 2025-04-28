"""
Convert the table with the research_group field into a dictionary.
The keys of the dictionary will be the subject id and the values will be the research_group value (i.e. the labels: AD, CN, MCI)
The table can be obtained in csv format from the ADNI website (https://ida.loni.usc.edu)
From the websitem, using the ADNI constructor you could create a filter with each one of the the research_group and then download the csv file with the subject id from the Download/Tables page.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import pandas
import json

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_label_AD = './data/ADNI_Labels/AD.csv'
path_label_MCI = './data/ADNI_Labels/MCI.csv'
path_label_CN = './data/ADNI_Labels/CN.csv'

label_to_int = dict(
    CN = 0,
    AD = 1,
    MCI = 2,
)

path_list = [path_label_AD, path_label_MCI, path_label_CN]

# Read and merge dataframes
df_list = []
for path in path_list:
    df = pandas.read_csv(path)
    df_list.append(df) 
df = pandas.concat(df_list, ignore_index=True)

dict_to_save_str = dict()
dict_to_save_int = dict()

for i in range(len(df)) :
    row = df.iloc[i]
    subj_id = row['subject_id']
    label = row['research_group']
    dict_to_save_str[subj_id] = label
    dict_to_save_int[subj_id] = label_to_int[label]

# Save dictionary
with open('./data/ADNI_Labels/all_labels_str.json', 'w') as f:  json.dump(dict_to_save_str, f)
with open('./data/ADNI_Labels/all_labels_int.json', 'w') as f:  json.dump(dict_to_save_int, f)




