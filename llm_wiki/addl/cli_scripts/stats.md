# stats.py

**Path:** `src/addl/cli_scripts/stats.py`

## Purpose
Backs two CLI commands: `compute-nii-dataset-stats` (via `addl.cli.compute_nii_dataset_stats`, which calls `compute_mean_and_std(args, file_dataset_extension=".nii")`) and `get-hdf5-dataset-length` (via `addl.cli.get_hdf5_dataset_length`, which calls `get_hdf5_dataset_length_func`). Both are dataset-introspection utilities: computing normalization statistics over a folder of sample files, and reporting how many samples an HDF5 dataset contains.

## Public API
### `compute_mean_and_std(args, file_dataset_extension: str = None) -> None`
- Supported `file_dataset_extension` values: `.npy`, `.pt`, `.pth`, `.nii` (passed in by the caller, not read from `args`).
- `args` fields used: `dataset_folder` (checked for existence, but see Notes — the actual file listing call uses `args.files_folder`, a mismatch), `filter` (optional substring filter), `debug`.
- Lists matching files via `addl.dataset.support_dataset.get_all_files_from_path`, treating **each file as one sample** (so this is meant for datasets stored as one-sample-per-file, not a single packed HDF5/`.nii` 4D volume — the caller is responsible for the per-sample-file layout being consistent).
- For each file, delegates to the private helper `__load_sample_file_and_compute_individual_stats` to get that sample's sum, sum-of-squares, and element count, then accumulates these globally.
- Computes `final_mean = sum/N` and `final_std = sqrt(sum_sq/N - mean**2)` and writes them as **plain text** (`str(value)`) to `mean.txt` and `std.txt` inside `args.files_folder`.
- No `compute_stats`-style per-sample normalization is applied here; this is purely a read-and-accumulate pass.

### `__load_sample_file_and_compute_individual_stats(path: str, extension: str)` (module-private, name-mangled by the double underscore only inside a class — here it's a module-level function, so the `__` prefix has no special Python semantics, it's just a naming convention for "internal")
- `.npy`/`.npz`/`.nii`: loads via `np.load` or `nib.load(...).get_fdata()`, computes stats with NumPy.
- `.pt`/`.pth`: loads via `torch.load(path)` (no `map_location`, so it respects whatever device the tensor was saved on), computes stats with Torch ops then converts to Python scalars via `.item()`.
- Returns a 3-tuple `(channel_sum, channel_squared_sum, total_data_point_per_sample)`.

### `get_hdf5_dataset_length_func(path: str, group_name: str = "data") -> int`
- Opens the HDF5 file at `path` read-only and returns `f["data"].shape[0]` — i.e. **the dataset name actually read is hardcoded to `"data"`**, ignoring the `group_name` parameter entirely (see Notes). Matches the fixed `"data"` dataset name used by `nii_to_hdf5_func` and `dataset_HDF5`.
- Has an inline `# TODO : Consider if move it inside the dataset module` — this logic plausibly belongs next to `addl.dataset.dataset_HDF5` rather than in the CLI layer.

## Notes
- **Bug:** `compute_mean_and_std` validates `args.dataset_folder` (existence check) but then calls `support_dataset.get_all_files_from_path(path_to_explore=args.files_folder, ...)` — a different attribute. Since the CLI parser for `compute-nii-dataset-stats` (in `cli.py`) only defines `--dataset_folder` (not `--files_folder`), calling this command as currently wired will raise `AttributeError: 'Namespace' object has no attribute 'files_folder'`. Also used again later for the output file paths (`args.files_folder`). This needs a fix (likely: replace both `args.files_folder` with `args.dataset_folder`) before the CLI command is usable.
- **Bug:** `get_hdf5_dataset_length_func`'s `group_name` parameter is accepted but unused — the code always reads `f["data"]`, so a non-default `group_name` silently has no effect.
- `__load_sample_file_and_compute_individual_stats` imports `nibabel`/`torch` lazily inside its branches even though `nibabel` is already imported at module level — redundant but harmless.
