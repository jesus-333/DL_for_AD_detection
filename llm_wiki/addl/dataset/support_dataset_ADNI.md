# support_dataset_ADNI.py

**Path:** `src/addl/dataset/support_dataset_ADNI.py`

## Purpose
ADNI-dataset-specific counterpart to `support_dataset_kaggle.py`: functions to discover, label, partition, and preprocess data coming from the ADNI (Alzheimer's Disease Neuroimaging Initiative) MRI dataset once it has been converted to PNG slices or packed into a tensor file. Split out of `support_dataset.py` as that file grew ("Due to the increase in size..."). This is the module most directly tied to the project's actual research data source (per the repo's stated scope: MRI-based AD detection). Depends on `dataset.py` (`MRI_dataset`) and `support_dataset.py`.

## Public API

### `get_dataset(list_of_path_to_data, n_samples=-1, merge_AD_class=0, print_var=True, seed=None) -> (file_path_list, label_list_int, label_list_str)`
PNG-path-based loader, ADNI analogue of `support_dataset_kaggle.get_data`. `list_of_path_to_data` is a list of per-class folder paths (author's own folder convention, e.g. `./data/ADNI_MRI_2D_Axial_png/{AD,CN,MCI}/`). `merge_AD_class`: `0` = keep CN/AD/MCI distinct, `1` = merge MCI into AD (binary CN-vs-rest). `n_samples > 0` triggers **balanced** subsampling via `create_ADNI_partition` (requires `n_samples >= number_of_classes`, else raises).

### `create_ADNI_partition(file_path_list, label_list_int, label_list_str, n_samples, seed=None) -> (sampled paths, sampled int labels, sampled str labels)`
Draws `n_samples // n_classes` samples per class without replacement; raises `ValueError` if any class has fewer available samples than needed. If `n_samples` isn't evenly divisible by the class count, tops up the shortfall with extra random samples drawn **without** excluding already-selected ones — the code comments acknowledge a known (accepted) duplicate-sample risk in that top-up step, described as negligible for small top-ups relative to dataset size.

### `get_dataset_V2(dataset_config, percentage_split_train_val=1, idx_to_use=None, seed=-1, preprocess_functions=None) -> (MRI_train_dataset, MRI_validation_dataset_or_None, seed)`
The main entry point used by tensor-file-based training (centralized and federated). Reads a pre-converted tensor file (`torch.load(..., mmap=True)`) plus a `dataset_info.csv` (must have `labels_int`/`labels_str` columns) from `dataset_config['path_data']`.
- Required `dataset_config` keys: `name_tensor_file`, `path_data`, `merge_AD_class`, `use_normalization`, `apply_rescale`, `rescale_factor`, `use_rgb_input`.
- `seed <= 0` triggers sampling a fresh random seed (note the default is `-1`, i.e. **random by default**, not reproducible unless the caller explicitly passes a positive seed) — used only for the train/val split (`support_dataset.get_idx_to_split_data_V3`), not for anything else in this function.
- `use_rgb_input=True` + single-channel data: automatically repeats the channel 3x (`data.repeat(1, 3, 1, 1)`); `use_rgb_input=True` + data with neither 1 nor 3 channels raises `ValueError`.
- `idx_to_use`: subsets the full tensor before splitting train/val — this is the hook federated-learning client code uses to restrict a client to its own partition of a larger shared tensor file (see `federated/` and the `create_idx_files_for_federated_simulations*.py` scripts).
- `use_normalization=True` and no explicit `preprocess_functions` passed: loads precomputed `dataset_mean.pt`/`dataset_std.pt` from `path_data` and builds a `Normalize` transform from them. If you *do* pass `preprocess_functions` yourself, this automatic normalization is skipped entirely — you are responsible for including normalization in what you pass.
- `apply_rescale=True`: divides the tensor by `rescale_factor` before building `MRI_dataset` — used when data was stored as e.g. `uint16` to save disk space and must be rescaled to `[0,1]` floats at load time.
- Returns `MRI_validation_dataset=None` when `percentage_split_train_val == 1` (i.e. "use everything for training, no validation split").
- Docstring has an open `TODO`: move preprocessing-function construction fully outside this function (currently it has smuggled-in normalization-building logic rather than only accepting a ready-made transform).

### `get_depth_map_order_single_sample_from_files_list(files_list) -> np.ndarray`
Parses the `idx_name.png` filename convention (produced by `scripts_python/data_manipulation/convert/all_subjects_ADNI_2D_MRI_V4_2.py` and later versions) to recover each slice's position in a 3D volume (the integer prefix before the first `_`).

### `get_depth_map_order_all_dataset(folders_paths_dict) -> dict[str, np.ndarray]`
Applies the above across every folder in a `paths_dict` (per-folder = per-subject-scan), producing the `depth_map_order_dict` consumed by `dataset_png.MRI_3D_dataset`.

### `get_labels_list_from_path_dict_V4_2(folders_paths_dict, subj_to_label_dict) -> np.ndarray`
Derives each folder's subject ID via `folder.split("__")[0].split("/")[-1]` (i.e. the folder/sample naming convention is `<subj_id>__<anything>`), looks it up in `subj_to_label_dict`. Raises `ValueError` (with a slightly malformed error message, printing a tuple-like concatenation) if a subject ID isn't found.

### `get_labels_dict_from_path_dict_V4_2(folders_paths_dict, subj_to_label_dict) -> dict`
Same idea but returns a dict keyed by folder, and **swallows** missing-subject errors with a bare `except:` + warning print instead of raising — inconsistent error-handling compared to the sibling `_list_` version above; be aware a missing subject here silently drops that folder rather than failing loudly.

### `get_preprocess_functions_ADNI_3D_png(input_size, use_normalization=False, z_matrix=None, slice=None) -> transforms.Compose`
Builds a resize(+normalize) transform for 3D-PNG ADNI data. Only `input_size == 176` is actually supported (176... `input_size == 224` immediately raises `ValueError`, appearing to be a placeholder for a case not yet implemented). Normalization stats are hardcoded **only** for `z_matrix == 44, slice == 4` — any other `(z_matrix, slice)` combination raises `ValueError`; this function is effectively single-configuration right now and will need extending for new volume shapes.

### `filter_AD_data(labels_list, class_to_keep, return_boolean_index=False) -> np.ndarray`
General-purpose filter: returns either integer indices or a boolean mask selecting samples whose label is in `class_to_keep`. Strictly validates that `labels_list`/`class_to_keep` elements are all `str` or all `int` and of matching type between the two lists (raises otherwise) — mixing e.g. string labels with an int `class_to_keep` will raise rather than silently failing to match.

### `merge_AD_class_function(labels_int, labels_str, merge_AD_class) -> labels_int`
Three merge policies for the finer-grained ADNI label set (`CN, AD, MCI, EMCI, LMCI, SMC`):
- `0`: no-op.
- `1`: binary CN vs. everything else (AD/MCI/EMCI/LMCI/SMC all → 1).
- `2`: 4-class CN/AD/MCI-group/SMC (EMCI and LMCI folded into MCI's integer).
This is distinct from (and more granular than) the `merge_AD_class` parameter in `get_dataset`/`create_ADNI_partition` above, which only knows about `CN/AD/MCI` (3 raw classes) — **the two `merge_AD_class` parameters in this same file are not semantically interchangeable**, double check which code path (PNG-based `get_dataset` vs. tensor-based `get_dataset_V2`) is in use before assuming a given integer value means the same thing.

## Notes
- This file mixes two generations of ADNI handling: an older PNG-path based flow (`get_dataset`, `create_ADNI_partition`) and a newer tensor-file based flow (`get_dataset_V2`) — the latter is the one wired into current centralized/federated training scripts per the `training`/`training_FL` script folders.
- Multiple functions are explicitly tied to specific folder-naming/file-naming conventions produced by specific versions of the `scripts_python/data_manipulation/convert/*` scripts (V4_2 in particular) — if those conversion scripts change their naming scheme, these parsing functions must be updated in lockstep.
