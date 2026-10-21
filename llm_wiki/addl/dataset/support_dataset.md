# support_dataset.py

**Path:** `src/addl/dataset/support_dataset.py`

## Purpose
Generic (dataset-agnostic) helper functions shared by the ADNI- and Kaggle-specific support modules (`support_dataset_ADNI.py`, `support_dataset_kaggle.py`): index/train-val-test splitting utilities, filesystem traversal helpers, label parsing from folder-structure conventions, and a couple of preprocessing-transform builders. The module docstring explicitly notes some functions are tied to the author's own folder layout, so they are not fully dataset-structure-agnostic despite living in the "generic" file.

## Public API

### `get_idx_to_split_data_V1(n_elements, percentage_split, seed=None) -> (idx_1, idx_2)`
Random permutation of `range(n_elements)` split two ways by `percentage_split`. Deterministic if `seed` given (sets `np.random.seed` globally — not a scoped RNG, so it affects global numpy random state).

### `random_split_array(input_array, percentage_split, seed=None) -> (part_1, part_2)`
Splits an arbitrary array two ways; special-cases `percentage_split == 0` (returns `[], input_array`) and `== 1` (returns `input_array, []`) to avoid calling `get_idx_to_split_data_V1` with an invalid 0/1 split. Note the second positional parameter in the signature is written `seed : None` (not `seed : int = None`) — there is no default value, so **this parameter is required** despite looking like a type hint with a default.

### `get_idx_to_split_data_V2(n_elements, percentage_split_list, seed=None) -> list[np.ndarray]`
N-way split (not just 2-way) via repeated calls to `random_split_array`. Validates `sum(percentage_split_list) == 1` with a small whitelist of float-precision-safe values near 1.0 — passing a list whose sum differs from 1 by more than that tolerance raises `ValueError`.

### `get_idx_to_split_data_V3(labels_list, percentage_split_list, seed=None) -> list[np.ndarray]`
Stratified version of V2: splits indices N ways while preserving per-label proportions in each split (runs V2 independently within each unique label's index subset, then concatenates). This is the function `support_dataset_ADNI.get_dataset_V2` uses to build its train/validation split while keeping class balance.

### `get_all_files_from_path(path_to_explore, filetype_filter=None) -> list[str]`
Recursive `os.walk`-based file listing, optionally filtered by filename suffix (e.g. `'png'`, `.nii`). Suffix check is plain `str.endswith`, so `filetype_filter='png'` also matches a file literally named `*.png` as well as anything merely ending in the substring `png` (no `.` is prepended automatically).

### `get_all_files_from_path_divided_per_folder(path_to_explore, filetype_filter=None) -> dict[str, list[str]]`
Groups the flat file list from `get_all_files_from_path` by containing folder. Used to build the `paths_dict` consumed by `dataset_png.MRI_3D_dataset`.

### `convert_label_from_str_to_int(labels_str_list, convert_MCI_to_AD=False) -> np.ndarray`
Hardcoded 'CN' → 0, 'AD' → 1, 'MCI' → 1 or 2 depending on `convert_MCI_to_AD`. Only recognizes these three label strings — anything else silently stays `0` (no explicit `else` branch/validation), since it initializes a zero array and only overwrites matched entries.

### `get_all_files_from_path_divided_per_folder` → see above.

### `convert_3D_matrix_to_image(data, path_to_save, file_name, axis_to_save, min_idx_to_save=0, max_idx_to_save=-1)`
Writes a 3D array out as a sequence of 2D PNGs along a chosen axis (0/1/2) using `cv2.imwrite`, creating `path_to_save` if needed. `max_idx_to_save=-1` means "all slices".

### `get_labels_from_path_list(file_paths_list) -> list[str]`
**Folder-structure-specific**: extracts the label by taking `file_path.split('/')[3]` — i.e. assumes a path of the form `./data/<dataset_name>/<label>/...`, with the label always at index 3 after splitting on `/`. Will silently return wrong labels if the path doesn't start with `./` or doesn't match this exact depth.

### `get_preprocess_function(dataset_name, input_size, grey_scale_image) -> (transforms.Compose, mean, std)`
Only supports `dataset_name == 'kaggle'`; hardcoded, precomputed mean/std constants for `input_size` 224 or 176 (computed previously via `scripts_python/analysis/compute_avg_std_dataset_2D_png.py`). Any other `dataset_name` or `input_size` raises `ValueError`. Duplicates similar hardcoded-stats logic found in `support_dataset_kaggle.get_dataset_with_preprocess_function_from_data` — if these constants ever need updating, both places must be kept in sync.

## Notes
- Several functions (`get_labels_from_path_list`, parts of `convert_3D_matrix_to_image` usage) hardcode assumptions about the author's own `./data/...` directory layout; porting to new data sources requires re-deriving these.
- `random_split_array`'s `seed` parameter having no real default is a likely footgun — check call sites before assuming `seed` is optional there.
- No class definitions in this file — purely functions, used as a shared utility module imported by `support_dataset_ADNI.py` and `support_dataset_kaggle.py` (`from . import dataset, support_dataset`).
