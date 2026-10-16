# nii_to_hdf5.py

**Path:** `src/addl/cli_scripts/nii_to_hdf5.py`

## Purpose
Backs the `nii-to-hdf5` CLI command (registered in `addl.cli.nii_to_hdf5`). It walks a folder tree for `.nii` files and packs all of them into a **single** HDF5 file (`dataset.h5`) containing one dataset named `"data"`, shaped `(num_samples, X, Y, Z)`. Optionally it also builds a companion `dataset_info.csv` from a user-supplied labels CSV, applies a resize (trilinear interpolation) and/or per-sample normalization before writing, and can compute/save the dataset-wide mean and std as `.npy` files. This is the main on-disk-dataset-prep path from raw ADNI NIfTI volumes toward the package's `dataset_HDF5` loader (see `llm_wiki/addl/dataset/dataset_HDF5.md`).

Hard dependencies on `h5py` and `nibabel` are enforced at import time via `try/except ImportError` (raises immediately with an instruction to install the library if missing).

## Public API
### `nii_to_hdf5_func(args) -> None`
- `args` is an `argparse.Namespace` from the CLI (fields: `dataset_folder`, `filter`, `output_folder`, `labels_file`, `reshape_size`, `minmax`, `z_score`, `compute_stats`, `debug`).
- Validates `dataset_folder` exists, lists all `.nii` files recursively via `addl.dataset.support_dataset.get_all_files_from_path`, and optionally filters the list to names containing `args.filter`.
- **Labels file** (`--labels_file`, optional): must be a `.csv` with required columns `file_name` and `labels_int`; optional columns `labels_str` and `subj_id` are picked up if present. Built into an in-memory `{basename: {label_int, [label_str], [subj_id]}}` dict keyed by the nii file's basename. If omitted, no `dataset_info.csv` is produced.
- **Reshape** (`--reshape_size`, optional): parsed from a string like `"(128, 128, 128)"` into a 3-tuple of positive ints; raises `ValueError` on a malformed or non-positive tuple. When set, every sample is resized via `torch.nn.functional.interpolate(..., mode="trilinear", align_corners=False)` (torch is imported lazily, only if this path is used). When *not* set, every sample's raw shape must exactly match the first sample's shape — a mismatch raises `ValueError` (no reshape ⇒ no shape coercion).
- **Normalization**: `--minmax` (per-sample min-max to [0,1]) and `--z_score` (per-sample zero-mean/unit-std) are independent flags; both can be applied, in which order (minmax first) matters if both are set — this is likely not intended for simultaneous use.
- Determines the HDF5 chunk shape as `(1, *sample_volume_shape)` so each chunk holds exactly one sample; compression is fixed to `"lzf"`.
- Output: writes `dataset.h5` to `--output_folder` (created if missing) or, if not given, back into `dataset_folder`. If labels were loaded, also writes `dataset_info.csv` with columns `['subj_id', 'labels_int', 'labels_str', 'file_name']` (missing optional fields filled with `"-"`).
- `--compute_stats`: accumulates global sum / sum-of-squares / voxel count across all samples (post-normalization, if normalization was applied) and writes `dataset_mean.npy` / `dataset_std.npy` into the output folder.
- `--debug`: periodic progress prints (roughly every 5% of files, or every 2 files for small datasets).

## Notes
- All samples are cast to `float32` regardless of source dtype.
- The function assumes every `.nii` file name in the labels CSV's `file_name` column matches a basename found under `dataset_folder` — a labels entry for a file that wasn't found in `nii_files_list` is silently never used, but a nii file *without* a labels entry will raise a `KeyError` on `labels_dict[file_name]` inside the loop (no explicit guard for that case).
- `dataset_info_csv` is built by repeated `pd.concat` inside the main loop — O(n²)-ish and could be slow for very large datasets; a list-of-rows + single `pd.DataFrame(...)` at the end would be more efficient, but this isn't in scope to fix here.
- `reshape_size` is applied before normalization, as documented in the CLI help text.
