# dataset_HDF5.py

**Path:** `src/addl/dataset/dataset_HDF5.py`

## Purpose
Implements the `MRI_dataset_HDF5` class: a `torch.utils.data.Dataset` that reads 3D MRI volumes lazily from an HDF5 file rather than holding everything in memory (contrast with `dataset.py`'s `MRI_dataset`, which requires the full tensor in RAM, or `MRI_dataset_mmap`, which uses a memory-mapped `.pt` file). This is the dataset class to use for datasets too large even for memory-mapping, or when the on-disk format produced by the `nii-to-hdf5` CLI command (`cli_scripts/nii_to_hdf5.py`) is the chosen storage format. Hard dependency on `h5py`; the module raises `ImportError` with a clear message at import time if it's missing.

## Public API

### `MRI_dataset_HDF5(torch.utils.data.Dataset)`
- `__init__(path_data, labels, idx_to_use=None, print_var=True)`: validates the file exists (`os.path.isfile`) and is a valid HDF5 file (`h5py.is_hdf5`), raising `FileNotFoundError`/`ValueError` respectively. Stores the **path only** — the file handle (`self.hdf5_file`) is intentionally left `None` ("opened lazily") because HDF5 file handles cannot be shared safely across `DataLoader` worker processes; each worker opens its own handle on first access.
  - `idx_to_use`: optional index remapping — when set, external index `i` passed to `__getitem__` is translated to `idx_to_use[i]` before reading the HDF5 chunk. This is the mechanism used to carve train/val/test (or federated client) subsets out of one shared HDF5 file without copying data.
- `__len__`: returns `len(idx_to_use)` if set, else `len(labels)`.
- `__getitem__(idx)`: opens `h5py.File(self.path_data, "r")` on first call (per worker/process), reads exactly one chunk from the `"data"` dataset at the (possibly remapped) index, and returns `(torch.from_numpy(volume), label)`. Comments note the HDF5 file **must** be chunked with chunk shape == sample shape for this single-chunk read to be efficient — if the file was written with different chunking, each `__getitem__` call may read much more data than needed.
  - A commented-out `swmr=True` option exists for single-writer-multiple-reader mode; currently disabled.
- `check_single_sample_V2(idx, axis="axial", figsize=None)`: interactive slider visualization, copied/adapted from `dataset.py`'s equivalent method with an added `axis` parameter (`"axial"`, `"sagittal"`, `"coronal"`) to select which plane is sliced when scrubbing through the volume via the index axis of `image[idx]`, `image[:, idx, :]`, or `image[:, :, idx]` respectively. Docstring flags the NIfTI axis convention (X=sagittal, Y=coronal, Z=axial) as `(TO CHECK)` — i.e. unverified by the author.

## Notes
- Data shape expected inside the HDF5 `"data"` dataset: one 3D volume (`X x Y x Z`) per chunk/index; labels are supplied separately (not stored inside the HDF5 file) and must already be aligned 1:1 with the sample order in the file.
- No `move_data_and_labels_to_device` method (unlike `dataset.py`'s classes) — by design, since the whole point is to avoid materializing all data; labels would need to be moved manually if needed.
- Companion CLI tooling: `nii-to-hdf5` (`cli_scripts/nii_to_hdf5.py`) creates files compatible with this class; `get-hdf5-dataset-length` (`cli_scripts/stats.py`) reports `len()` without instantiating the full dataset.
