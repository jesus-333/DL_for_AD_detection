# pt_to_npy.py

**Path:** `src/addl/cli_scripts/pt_to_npy.py`

## Purpose
Backs two CLI commands that are mirror images of each other: `pt-to-npy` (converts `.pt`/`.pth` torch tensor files to `.npy`) and `npy-to-pt` (the reverse, `.npy` to `.pt`). Both are implemented by the single function below, toggled by a `reverse` flag — `addl.cli.npy_to_pt` calls `pt_to_npy.pt_to_npy(args, reverse=True)` and `addl.cli.pt_to_npy` calls `pt_to_npy.pt_to_npy_func(args)` (note: `cli.py`'s `pt_to_npy` command actually calls `pt_to_npy.pt_to_npy_func`, not this module's `pt_to_npy` function directly — see Notes).

## Public API
### `pt_to_npy(args, reverse: bool = False) -> None`
- `args` fields: `files_folder` (required), `filter` (optional substring match on basename), `output_folder` (optional, defaults to `files_folder`), `debug`.
- Recursively collects source files via `addl.dataset.support_dataset.get_all_files_from_path`: when `reverse=True` it looks for `*.npy`; when `reverse=False` it looks for both `*.pt` and `*.pth` and concatenates the two lists.
- Raises `ValueError` if `files_folder` is missing/doesn't exist, or if no matching files are found (before/after filtering).
- Conversion per file:
  - `reverse=True` (npy → pt): `np.load` then `torch.from_numpy(...)`, saved via `torch.save` with extension swapped `.npy` → `.pt`.
  - `reverse=False` (pt/pth → npy): `torch.load(path, map_location='cpu').numpy()`, saved via `np.save` with extension swapped `.pth`→`.npy` or `.pt`→`.npy` (checked via substring `'pth' in path_current_file`, not a suffix check).
- `--debug`: prints progress roughly every 5% of files (or every 2 for small file counts) — note `n_elements_to_print` is only computed inside the `if args.debug:` block, so debug printing logic depends on that variable existing only when debug is on (consistent, but fragile if refactored).

### `pt_to_npy_func(args) -> None`
Referenced from `cli.py` (`from .cli_scripts import pt_to_npy; pt_to_npy.pt_to_npy_func(args)`) but **this name is not defined anywhere in this file** — only `pt_to_npy(...)` exists. This is a latent bug: invoking the `pt-to-npy` CLI command will raise `AttributeError: module 'addl.cli_scripts.pt_to_npy' has no attribute 'pt_to_npy_func'`.

## Notes
- **Bug to flag/fix in future work:** `cli.py`'s `pt_to_npy()` CLI entry point calls `pt_to_npy.pt_to_npy_func(args)`, which doesn't exist in this module. Either `cli.py` should call `pt_to_npy.pt_to_npy(args)` (equivalent to `reverse=False`, the default), or this module should define a `pt_to_npy_func` alias/wrapper. The `npy-to-pt` command works correctly since it calls `pt_to_npy.pt_to_npy(args, reverse=True)`.
- `torch.load(..., map_location='cpu')` assumes the saved tensor is a plain tensor with a `.numpy()` method — will fail if a `.pt` file contains something else (e.g. a state dict or non-tensor object).
- Depends on `h5py`/`nibabel` import guards at module load time even though this file doesn't use either library directly (likely copy-paste residue from `nii_to_hdf5.py`); they still need to be installed for this module to import successfully.
