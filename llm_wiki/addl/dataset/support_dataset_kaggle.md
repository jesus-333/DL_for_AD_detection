# support_dataset_kaggle.py

**Path:** `src/addl/dataset/support_dataset_kaggle.py`

## Purpose
Support functions specific to the public Kaggle "Alzheimer MRI 4 classes" dataset (https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset), used as an easily-obtainable public benchmark alongside (or instead of) the private/gated ADNI dataset handled by `support_dataset_ADNI.py`. Split out of `support_dataset.py` for the same reason as the ADNI module. Depends on `dataset_png.py` (`MRI_2D_dataset`) and `support_dataset.py`.

## Public API

### `get_data(path_files_Moderate_Demented, path_files_Mild_Demented, path_files_Very_Mild_Demented, path_files_Non_Demented, merge_AD_class=0, print_var=True) -> (file_path_list, label_list_int, label_list_str)`
Reads all files from the four fixed class folders (the Kaggle dataset's four severity classes) and returns parallel path/int-label/str-label arrays.
- `merge_AD_class`: `0` = keep all 4 classes; `1` = merge all demented classes into one (binary Non-Demented vs. Demented); `2` = merge only Mild and Very-Mild (keep Moderate separate) — a 3rd distinct scheme from `support_dataset_ADNI`'s `merge_AD_class` semantics (see cross-reference note in that module's doc) — **do not assume these integers mean the same thing across the two "kaggle" and "ADNI" modules.**
- **Caution — parameter order vs. positional call risk**: the signature's parameter order is `(Moderate, Mild, VeryMild, NonDemented)`, but inside the function body `download_and_get_data` (below) calls `get_data(path_files_Non_Demented, path_files_Mild_Demented, path_files_Moderate_Demented, path_files_Very_Mild_Demented, ...)` — i.e. it passes `Non_Demented` into the `path_files_Moderate_Demented` slot and `Moderate_Demented` into the `path_files_Mild_Demented` slot. Because `get_data` is called positionally and internally only uses these arguments to call `support_dataset.get_all_files_from_path` (which doesn't care about the label at this stage — labels come later from `get_labels_from_path_list`, which reads the real folder name off the path, not off which parameter slot was used), this is **very likely harmless in practice** since the label assignment doesn't depend on argument order, but it means the local variable name `path_files_X` passed to a given parameter slot does NOT correspond to that parameter's name — worth fixing for clarity, and worth being careful with if this function's internals are ever changed to trust the parameter identity rather than re-deriving labels from the path string.

### `download_and_get_data(path_to_save_data='./data/', merge_AD_class=False, print_var=True) -> same return as get_data`
Downloads the dataset via `kagglehub.dataset_download(...)` if not already present at `path_to_save_data`, moves it into place with `shutil.move`, then delegates to `get_data` (see the parameter-order note above — this is the actual call site exhibiting that behavior). Requires the optional `kagglehub` package; if missing, the module prints a warning at import time and this function raises `ValueError` if called. Note the default type hint on `merge_AD_class` here is `bool` but the parameter is really used as the same 0/1/2 int enum as `get_data` — the type hint is misleading/stale.

### `get_dataset_with_preprocess_function_from_data(file_path_list, label_list_int, model_config, dataset_config) -> dataset_png.MRI_2D_dataset`
Builds an `MRI_2D_dataset` with a preprocessing pipeline selected by `model_config['input_size']` (224 or 176 only, else raises `ValueError`), using the **same hardcoded Kaggle mean/std constants** that also appear in `support_dataset.get_preprocess_function` — these two implementations have duplicated, independently-maintained copies of the same magic numbers; if the dataset stats are ever recomputed, both places need updating.
- `dataset_config` required keys: `grey_scale_image`, `load_data_in_memory`.
- `model_config` required key: `input_size`.

## Notes
- No tensor-file / mmap / HDF5 path for Kaggle data in this module — Kaggle data stays PNG-path-based via `dataset_png.MRI_2D_dataset`, unlike the ADNI side which has evolved a tensor-file fast path (`support_dataset_ADNI.get_dataset_V2`).
- The duplicate hardcoded normalization constants (here and in `support_dataset.get_preprocess_function`) are a maintenance risk — consider consolidating if touching either.
