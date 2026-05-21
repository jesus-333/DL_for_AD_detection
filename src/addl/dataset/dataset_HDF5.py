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
    """
    Class to load MRI data from HDF5 files. The HDF5 file is organized in chunks, where each chunk corresponds to a sample (i.e., a 3D volume).
    This allows to efficiently load large datasets that do not fit in memory, as only the required chunk is loaded when the __getitem__ method is called.

    Parameters
    ----------
    path_data : str
        Path to the HDF5 file containing the data.
    labels : list
        List of labels corresponding to the samples in the HDF5 file. The length of this list must be equal to the number of samples in the HDF5 file (i.e., the number of chunks).
    idx_to_use : list, optional
        List of indices of the samples to use. If None, all samples are used.
        If used, when you call the __getitem__ method with an index i, the sample returned will be the one corresponding to the index idx_to_use[i] in the HDF5 file. Default is None.
        This option was added to allow the use of a subset of samples with the HDF5 format (e.g. for training/validation split)
    print_var : bool, optional
        If True, print some variables to check that the dataset is loaded correctly. Default is True.
    """

    def __init__(self, path_data : str, labels : list, idx_to_use : list = None, print_var : bool = True) :

        # Check that the HDF5 file exists and is valid
        if not os.path.isfile(path_data) :
            raise FileNotFoundError(f"HDF5 file not found at path {path_data}. Please check the path and try again.")
        elif not h5py.is_hdf5(path_data) :
            raise ValueError(f"The file at path {path_data} is not a valid HDF5 file. Please check the file and try again.")

        # Store the path to the HDF5 file. The file will be opened lazily when the __getitem__ method is called for the first time.
        self.path_data = path_data
        if print_var : print(f"HDF5 file found at path {path_data} and is valid. Path stored.")
        
        # Save labels
        self.labels = torch.asarray(labels)
        if print_var : print("Labels loaded.")
        
        # Save idx_to_use
        self.idx_to_use = idx_to_use
        
        # Note that the HDF5 file is not opened here, but it will be opened lazily when the __getitem__ method is called for the first time.
        # This is because the HDF5 file cannot be shared between different workers of the DataLoader, so each worker will have its own handle to the file, which will be opened when needed.
        self.hdf5_file = None # opened lazily per worker

    def __len__(self):
        if self.idx_to_use is not None :
            return len(self.idx_to_use)
        else :
            return len(self.labels)

    def __getitem__(self, idx):
        # Each DataLoader worker gets its own handle
        if self.hdf5_file is None : self.hdf5_file = h5py.File(self.path_data, "r", swmr = True)
        
        # (OPTIONAL) If idx_to_use is not None, get idx_to_use[idx] instead of idx.
        if self.idx_to_use is not None : idx = self.idx_to_use[idx]
        
        # Read the volume corresponding to the given index. Note that the HDF5 file is organized in chunks, so this operation reads exactly one chunk of data, which is efficient even for large datasets that do not fit in memory.
        # Note that to work properly, you must use chunk with the same shape of the samples.
        volume = self.hdf5_file["data"][idx]

        return torch.from_numpy(volume)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
