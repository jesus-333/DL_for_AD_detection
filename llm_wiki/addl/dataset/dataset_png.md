# dataset_png.py

**Path:** `src/addl/dataset/dataset_png.py`

## Purpose
Implements dataset classes that read individual image files (PNG by default, DICOM via a subclass) from disk paths, rather than from a pre-packed tensor (`dataset.py`) or HDF5 file (`dataset_HDF5.py`). This is the original/baseline dataset format for both the Kaggle AD dataset and ADNI-derived 2D-slice datasets, and is still what `support_dataset_kaggle.py` and parts of `support_dataset_ADNI.py`/the `scripts_python/data_manipulation` pipeline build on top of. It supports both 2D single-slice samples (`MRI_2D_dataset`) and 3D volumes reconstructed by stacking many per-slice PNGs in the correct order (`MRI_3D_dataset`).

## Public API

### `MRI_2D_dataset(torch.utils.data.Dataset)`
- Takes `path_list` (image file paths) + `labels_list`, loads images with `torchvision.io.read_image`, normalizes pixel values to `[0, 1]` by dividing by 255.
- `load_data_in_memory` (bool, constructor arg): if `True`, all images are loaded and stacked eagerly via `load_dataset()`; if `False`, each `__getitem__` call loads from disk on demand via `load_image`.
- `preprocess_functions`: a `torchvision.transforms.Compose` applied per-sample at access time (unlike `dataset.py`'s `MRI_dataset`, which applies preprocessing once at construction). Can be (re)assigned later via `set_preprocess_functions`, which **requires** a `Compose` instance (raises `ValueError` otherwise) — a plain callable is not accepted here, unlike the constructor's `preprocess_functions` arg which accepts anything callable.
- `grey_scale_image` (bool): selects `ImageReadMode.GRAY` vs `RGB` when reading.
- `add_extra_dimensions_to_single_sample` (bool, default `False`): when `True`, a single-sample `__getitem__` call returns shape `(1, 1, H, W)` instead of `(1, H, W)` — useful only outside a `DataLoader` (which already adds the batch dim), so leave `False` when using `DataLoader`.
- `load_image(path)`: handles both a single path (str) and an array/list of paths (used when a dataset slice `ds[i:j]` is requested), stacking multiple images via `torch.stack`. **Caveat documented in the code**: when `load_data_in_memory=False` and no `preprocess_functions` is set, all images in a batch/slice access must already share the same shape or `torch.stack` will fail.
- `load_dataset()` / `remove_dataset_from_memory()`: eager-load/unload all images into `self.data_loaded`.
- `visualize_sample(idx)`: quick matplotlib preview.
- `visualize_preprocess_effect(idx)`: side-by-side plot of raw vs. preprocessed image plus printed mean/std; requires `preprocess_functions` to be set and only supports the "not loaded in memory" path explicitly (`load_data_in_memory=True` branch is a no-op `pass`, so the "preprocessed" panel would show stale/undefined data if used that way — effectively this method does not fully support in-memory datasets).
- `compute_avg_and_std_for_dataset(n_samples=-1)`: computes per-channel mean/std over `n_samples` (or all, if `-1`), temporarily restricting `preprocess_functions` to only its `Resize`/`CenterCrop` transforms (if present) before computing stats, so the result matches what the final normalization will see spatially. Restores the original `preprocess_functions` afterward.
- `update_preprocess_functions_normalize()`: finds a `torchvision.transforms.Normalize` inside `preprocess_functions` (if present) and overwrites its `mean`/`std` in place using `compute_avg_and_std_for_dataset()`.

### `MRI_2D_dataset_dicom(MRI_2D_dataset)`
- Same interface, but `path_list` points to `.dcm` files; overrides `load_image` to read via `pydicom.dcmread(path).pixel_array`, clips pixel values `>255` down to `255` (flagged in a comment as a provisional decision, "In the future I will check if this is the right thing to do"), then normalizes to `[0, 1]`.
- Forces `grey_scale_image=True` via `super().__init__(..., grey_scale_image=True)`.
- `create_copy_of_image_for_depth_map(image)`: triplicates a single-channel image into a pseudo-RGB 3-channel tensor (all channels identical) so grayscale MRI slices can be fed to models expecting RGB input (e.g. VGG/Inception backbones in `model/vgg_nets.py`).
- Docstring explicitly states **"NOT USED FOR NOW"** — treat as an experimental/inactive code path.
- Requires `pydicom`; if not installed, the module prints a warning at import time (not an exception) and this class will fail at use-time instead of import-time.

### `MRI_3D_dataset(MRI_2D_dataset)`
- Reconstructs a 3D volume per sample by stacking a *folder's worth* of 2D PNG slices in the correct depth order.
- `__init__(paths_dict, depth_map_order_dict, labels_dict, load_data_in_memory=False, load_data_type=0, preprocess_functions=None)`: all three dict args are keyed by folder path (one folder = one sample/subject scan); `paths_dict[folder]` lists slice file paths, `depth_map_order_dict[folder]` gives the permutation that sorts them into correct anatomical order, `labels_dict[folder]` gives that sample's label. Typical producers of these dicts: `support_dataset.get_all_files_from_path_divided_per_folder` (paths) and `support_dataset_ADNI.get_depth_map_order_all_dataset`/`get_labels_dict_from_path_dict_V4_2` (order/labels) — i.e. this class is wired specifically to the output of the `convert/all_subjects_ADNI_*` conversion scripts in `scripts_python/data_manipulation/convert/`.
- Forces `grey_scale_image=True` internally (3D MRI slices are single-channel).
- `load_data_type` (0 or 1, only used when `load_data_in_memory=True`): `0` keeps samples as a Python list of per-sample tensors (supports variable depth across samples); `1` stacks everything into one dense tensor (requires every sample to have the same number of slices).
- `__getitem__`/`load_sample`/`load_singe_sample` [sic — typo retained from source]: mirror the 2D class's slice-vs-single-index handling, but build each sample by loading and stacking all slice PNGs for that folder in `depth_map_order` order.
- `load_dataset(load_data_type=0)`: eager full-dataset load with progress printing every ~10% of samples.
- `check_single_sample_V1`/`check_single_sample_V2`: same depth-map visual sanity-check idea as `dataset.py`, specialized to call `load_sample` first.

## Notes
- `Attributes` docstring section of `MRI_3D_dataset` is literally `TODO` — attribute list not documented beyond what's inferable from `__init__`.
- There's a source-level typo `load_singe_sample` (missing "l") used consistently, so don't "fix" it without also updating all call sites.
- No HDF5/tensor-file awareness in this module — it is purely path-and-PNG based; pairs with `support_dataset.get_all_files_from_path`/`get_all_files_from_path_divided_per_folder` and `support_dataset_ADNI`/`support_dataset_kaggle` for building the `path_list`/`paths_dict`/`labels_dict` inputs.
