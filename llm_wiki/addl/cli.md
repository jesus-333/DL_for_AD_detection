# cli.py

**Path:** `src/addl/cli.py`

## Purpose

Implements the command-line entry points exposed by the installed package. Each public function here is wired as a console-script entry point in `pyproject.toml`'s `[project.scripts]` table, e.g.:

```toml
nii-to-hdf5 = "addl.cli:nii_to_hdf5"
pt-to-npy   = "addl.cli:pt_to_npy"
npy-to-pt   = "addl.cli:npy_to_pt"
get-hdf5-dataset-length = "addl.cli:get_hdf5_dataset_length"
wandb-get-run-status = "addl.cli:wandb_get_run_status"
wandb-download-project-runs-metrics = "addl.cli:wandb_download_project_runs_metrics"
```

Each function in this file follows the same pattern: build an `argparse.ArgumentParser` with `prog` and `description`, define the command's flags, parse `sys.argv`, then import the real implementation from `addl.cli_scripts` (lazily, inside the function body — this keeps startup of unrelated CLI commands cheap) and call into it. This file is intentionally thin: it owns argument parsing/help text, `addl/cli_scripts/*.py` owns the actual logic. See `llm_wiki/addl/cli_scripts/` for the implementation details of each command.

## Public API

### `nii_to_hdf5() -> None`
CLI command `nii-to-hdf5`. Converts all `.nii` files under `--dataset_folder` (recursively) into a single HDF5 file. Flags: `--dataset_folder`, `--filter` (substring filter on filename), `--output_folder`, `--labels_file` (CSV with `file_name`/`labels_int`, optional `labels_str`/`subj_id` — producing a `dataset_info.csv` alongside the HDF5 output when given), `--reshape_size` (tuple string, applied before normalization), `--minmax` / `--z_score` (mutually-relevant sample-wise normalization flags), `--compute_stats`, `--debug`. Delegates to `cli_scripts.nii_to_hdf5.nii_to_hdf5_func(args)`.

### `nii_to_pt() -> None`
Declared but **not implemented** — docstring is literally `"""TODO"""` and the function body is empty. Not wired in `pyproject.toml`'s `[project.scripts]` either (no matching entry). A no-op placeholder for a future command.

### `npy_to_pt() -> None`
CLI command `npy-to-pt`. Converts `.npy` files under `--files_folder` to `.pt` files. Flags: `--files_folder`, `--filter`, `--output_folder`, `--debug`. Note the implementation trick: it imports `cli_scripts.pt_to_npy` and calls `pt_to_npy.pt_to_npy(args, reverse=True)` — i.e. the npy→pt direction is implemented as the *reverse* mode of the same underlying function used for pt→npy, not a separate function.

### `pt_to_npy() -> None`
CLI command `pt-to-npy`. Converts `.pt`/`.pth` files under `--files_folder` to `.npy` files. Same flags as `npy_to_pt`. Delegates to `cli_scripts.pt_to_npy.pt_to_npy_func(args)` (note: a *different* function name than the one `npy_to_pt` calls — `pt_to_npy_func` vs `pt_to_npy`).

### `get_hdf5_dataset_length() -> None`
CLI command `get-hdf5-dataset-length`. Reads an HDF5 file (`--path_hdf5_file`, required) and prints the number of samples in the group named `--group_name` (default `"data"`). Delegates to `cli_scripts.stats.get_hdf5_dataset_length_func(path, group_name)` and prints the raw integer to stdout (meant to be captured by shell scripts).

### `compute_nii_dataset_stats() -> None`
**Not registered** as a `[project.scripts]` entry point (no matching line in `pyproject.toml`), even though it's a fully implemented function here — so today it's only reachable by calling `addl.cli.compute_nii_dataset_stats` directly, not via a shell command. Computes mean/std of a dataset of `.nii` files (`--dataset_folder`, `--filter`, `--debug`) and saves them as `.npy` files, via `cli_scripts.stats.compute_mean_and_std(args, file_dataset_extension=".nii")`.

### `wandb_get_run_status() -> None`
CLI command `wandb-get-run-status`. Given `--entity`, `--project`, `--run_id` (all required), prints the status of that W&B run. Delegates to `cli_scripts.wandb_scripts.get_run_status(args)`.

### `wandb_download_project_runs_metrics() -> None`
CLI command `wandb-download-project-runs-metrics`. Given `--entity`, `--project` (required) and optional `--path_save` (defaults to `wandb_metrics.json` in the cwd), downloads all metrics for all runs of a W&B project into a JSON file. Delegates to `cli_scripts.wandb_scripts.download_project_runs_metrics(args)`. Its own docstring is empty (`""" """"`).

## Notes

- CLI naming convention matches `coding_style_instructions_ENG.md` (now at `llm_wiki/coding_style_instructions_ENG.md`): multi-word CLI commands use hyphens (`nii-to-hdf5`, `get-hdf5-dataset-length`, ...).
- Asymmetry to watch for: `npy_to_pt()` and `pt_to_npy()` both live in `cli_scripts.pt_to_npy` but call two differently-named functions (`pt_to_npy(..., reverse=True)` vs `pt_to_npy_func(...)`) — easy to confuse when editing.
- `nii_to_pt` is an unimplemented stub; `compute_nii_dataset_stats` is implemented but not exposed as an installed command. Either could be intentionally incomplete/in-progress — don't assume dead code without checking with the user/history first.
- All argument parsing happens via `argparse.ArgumentParser` built fresh inside each function (not a shared subparser structure), so there is no single `addl --help` listing all commands — each is its own standalone console script per `pyproject.toml`.
