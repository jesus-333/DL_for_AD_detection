# `scripts_python/`

**Path:** `scripts_python/`

## What it is

Python scripts that are **not** part of the installable `addl` package but *use* it — training entry points, data analysis, data preprocessing/conversion, and throwaway test/debug code written over the course of the project. Per `scripts_python/README.md`, every script is meant to be run from the **repository root** (e.g. `python scripts_python/training/demnet_ADNI_wandb.py`); the author typically uses an IPython interactive shell from the repo root instead of plain `python`.

As the user has noted directly: this tree accumulates scripts written for many one-off purposes (training variants, analysis, preprocessing) that were never meant to graduate into the package — expect duplication, superseded versions (`_V1`, `_V2`, `_V3`...), and dead ends.

## Layout

- **`analysis/`** — scripts for scientific analysis, e.g. evaluating trained models (`evalute_trained_demnet_1.py`, `evalute_trained_demnet_2_transfer_learning.py`, `vgg_evaluate_accuracy.py`), comparing LR schedulers (`compare_lr_schedulers_V1.py` / `V2.py`), computing dataset statistics (`compute_avg_std_dataset_2D_png.py`, `compute_avg_std_dataset_3D.py`, `compute_max_dataset_3D.py`), DICOM inspection (`dicom_and_thresholding.py`, `dicom_stats_3D_dataset.py`, `dicom_stats_single_files.py`, `dicom_visualize_3D_sample.py`), data-distribution checks (`distribution_data_V1_ADNI.py`, `distribution_data_V1_kaggle.py`), and a federated-learning standard-deviation experiment (`federated_std_experiment.py`).

- **`data_manipulation/`** — scripts to manipulate raw/intermediate data (convert `.nii`/DICOM files, build label files, build federated client index files, interpolate volumes, etc.). Per its README: these assume data downloaded from the [ADNI website](https://adni.loni.usc.edu/data-samples/adni-data/) and organized as `data/<dataset_name>/<class>/...` (e.g. `data/ADNI_3_fMRI/MCI`), so they are **not** expected to run unmodified in a different data layout. Notable contents:
  - `create_ADNI_partition.py`, `create_labels_file_ADNI_V1.py` / `V2.py`, `create_labels_from_folders.py` — build dataset partitions/label files.
  - `create_idx_files_for_federated_simulations*.py` (3 versions) — generate the client index files consumed by the "V2" federated training scripts (see `scripts_sh.md` / `scripts_sh_V2.md`) — this is how train/val split and client partitioning became reproducible/decoupled from server startup.
  - `create_wandb_dataset_artifact_V1.py` / `V2.py`, `download_kaggle_dataset.py` — dataset acquisition/versioning helpers.
  - `interpolate_z_axis_*.py`, `merge_3D_dataset_tensor.py`, `change_plane_axial_to_sagittal.py` — volume-level preprocessing.
  - `convert/` — a dedicated subfolder (split out "since the functions that converted the data were starting to increase in number", per its README) holding the many successive versions of ADNI-to-image/tensor conversion scripts (`all_subjects_ADNI_2D_MRI_V1.py` through `V6_convert_nii.py`, plus single-sample/DICOM-unpacking variants). The version suffixes track the evolution of the preprocessing approach over time — check file dates/git history before picking one to build on.

- **`plot/`** — plotting scripts used to generate paper figures (`metrics_for_paper*.py`, `partial_leakage_*.py`, `plot_bias_demnet.py`) plus their generated output (`metrics_for_paper/` contains `.eps`/`.pdf`/`.png` figures and cached `.npy` arrays — these are build artifacts, not source).

- **`test_code/`** — scratch scripts used to test/debug code (dataset classes, read timing, memory usage, loading ADNI variants) — not a `pytest`/unit-test suite, just exploratory scripts (per its README).

- **`training/`** — centralized-training entry points (FL training has its own folder, `training_FL/`). Includes per-model training scripts (`demnet_ADNI_wandb_*.py`, `demnet_kaggle_wandb.py`, `demnet_centralized_V2.py`, `swin_transformer_centralized_V2.py`, `vav_centralized_V2.py`, `vit_transformer_centralized_V2.py`, `vgg_ADNI_wandb.py`, `vgg_kaggle*.py`) plus config-file maintenance utilities (`update_dataset_config.py`, `update_lr_scheduler.py`, `update_optimizer.py`, `update_training_config.py` (also writes the opt-in long-training keys: `--resume_from_backup`, `--start_epoch`, `--path_to_save_snapshot`, `--stop_file_path`, `--wandb_watch_log*`; each is written only if passed), `update_model_config_*.py` — one per model family — `reset_config_files.py`, `get_oldest_file.py`).

- **`training_FL/`** — Flower-based federated-learning applications, one self-contained Flower app per subfolder (its own `pyproject.toml`, `client_app.py`, `server_app.py`, `config/`):
  - `ADNI_demnet_fedavg_with_wandb/` — first iteration: train/val split and client data partitioning happen inside `prepare_data_for_FL_training` in `server_app.py` at server-instantiation time.
  - `ADNI_demnet_fedavg_with_wandb_V2/` — second iteration (per its README): instead consumes pre-built index files from `data_manipulation/create_idx_files_for_federated_simulations.py`. Its `config/` subfolder contains a very large number of per-run, auto-generated `dataset_<id>.toml` / `lr_scheduler_config_<id>.toml` / `model_<id>.toml` / `optimizer_config_<id>.toml` files (one set per experiment run, named by numeric job/run id) — these are run artifacts, not hand-maintained defaults; don't enumerate or rely on specific ones.

## Notes for future work

- Treat version suffixes (`_V1`, `_V2`, `_V3`...) as a changelog-in-filenames: the highest version number is usually (but not always confirmed) the most current approach — check README files and git history in the specific subfolder before building on an older one.
- `training_FL/*/config/*_<id>.toml` and `scripts_sh_V2/config/debug/*` are both auto-generated per-run artifacts — same pattern repeated in two places. If a future "run tracking" convention is introduced, it should probably unify these.
- None of this is covered by tests or meant to be imported as a library — scripts here call into the `addl` package but are not called by it.
