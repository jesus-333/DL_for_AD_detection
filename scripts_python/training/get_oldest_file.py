"""
Given a path, print the oldest file in it. 
If filter is provided, only files with the given filter in the file name are considered (e.g., filter='model' will only consider files with 'model' in the name).

Various :
- Note that the path is printed because this script is meant to be used in a bash script, where the output of this script will be captured and used as an input for another command.
- Note that the path will be printed from where the script is called (unless you pass the `return_only_file_name` argument, in which case only the file name will be printed).
- This script is here because it is used to get the model wieghts/optimizer state files that are used to resume training (see function backup_every_epoch in support_training.py for more info).

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Arguments

import argparse

parser = argparse.ArgumentParser(description = 'Get the oldest file in a given path.')
parser.add_argument('--path', type = str, help = 'The path to search for the oldest file.')
parser.add_argument('--filter', type = str, default = None, help = 'Only consider files with this filter in the name.')
parser.add_argument('--return_only_file_name', action = 'store_true', default = False, help = 'If set, only the file name is returned, otherwise the full path is returned.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import glob
# The Python glob module provides tools to find path names matching specified patterns that follow Unix shell rules. You can use this module for file and directory pattern matching.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_oldest_file(path : str, filter : str = None, return_only_file_name : bool = False) -> str :
    # Get all files in the path, optionally filtering by the provided filter
    if filter is not None:
        files = glob.glob(os.path.join(path, f'*{filter}*'))
    else:
        files = glob.glob(os.path.join(path, '*'))
    
    # If no files are found, return None
    if len(files) == 0:
        return None
    
    # Get the oldest file based on creation time
    # The key argument applies the os.path.getctime function to each file in the list, which returns the creation time of the file.
    # Therefore, min(files, key = os.path.getctime) will return the file with the smallest creation time, which is the oldest file.
    oldest_file = min(files, key = os.path.getctime)
    
    if return_only_file_name :
        return os.path.basename(oldest_file)
    else :
        return oldest_file

if __name__ == '__main__' :
    oldest_file = get_oldest_file(args.path, args.filter, args.return_only_file_name)
    if oldest_file is not None:
        print(oldest_file)
    else:
        print('-')
