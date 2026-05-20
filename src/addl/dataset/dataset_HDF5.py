"""
Implementation of the dataset class for HDF5 files. Used to load big datasets that do not fit in memory.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import torch

try :
    import h5py
except ImportError :
    raise ImportError("h5py library is not installed. Please install it to use the dataset_HDF5 class.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class MRI_dataset_HDF5(torch.utils.data.Dataset) :

    def __init__(self, path_data : str, labels : list, print_var : bool = True) :

        # Check that the HDF5 file exists and is valid
        if not os.path.isfile(path_data) :
            raise FileNotFoundError(f"HDF5 file not found at path {path_data}. Please check the path and try again.")
        elif not h5py.is_hdf5(path_data) :
            raise ValueError(f"The file at path {path_data} is not a valid HDF5 file. Please check the file and try again.")

        # Store the path to the HDF5 file. The file will be opened lazily when the __getitem__ method is called for the first time.
        self.path_data = path_data
        if print_var : print(f"HDF5 file found at path {path_data} and is valid. Path stored.")

        self.labels = torch.asarray(labels)
        if print_var : print("Labels loaded.")
        
        # Note that the HDF5 file is not opened here, but it will be opened lazily when the __getitem__ method is called for the first time.
        # This is because the HDF5 file cannot be shared between different workers of the DataLoader, so each worker will have its own handle to the file, which will be opened when needed.
        self.hdf5_file = None # opened lazily per worker

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # each DataLoader worker gets its own handle
        if self.hdf5_file is None :
            self.hdf5_file = h5py.File(self.path_data, "r", swmr = True)
        
        # Read the volume corresponding to the given index. Note that the HDF5 file is organized in chunks, so this operation reads exactly one chunk of data, which is efficient even for large datasets that do not fit in memory.
        # Note that to work properly, you must use chunk with the same shape of the samples.
        volume = self.hdf5_file["volumes"][idx]

        return torch.from_numpy(volume)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

