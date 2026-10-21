# `scripts_sh/` (original version)

**Path:** `scripts_sh/`

## What it is

The **original** collection of shell scripts used to launch training jobs, mostly written to run on an HPC cluster with a SLURM scheduler (the author's own HPC — University of Luxembourg's, see `scripts_sh/README.md`). These scripts are **not** part of the installable `addl` package — they are standalone entry points that call into it (training scripts, config scripts, debug scripts). A newer, improved iteration lives in `scripts_sh_V2/` (see `scripts_sh_V2.md`); this folder is the earlier version, kept around rather than deleted.

Scripts with a `_local` suffix can run on any machine; the rest assume `srun`/SLURM (removing the `srun` prefix from commands would make them locally runnable too, per the README).

## Layout

- `debug/` — scripts to debug/test the training code locally (`train_demnet_local*.sh`, `train_vgg_local*.sh`, `check_memory_usage.sh`), including FL and "parallel" (multiple simultaneous runs) variants.
- `create_config/` — scripts that generate the TOML config files used by training runs (one script per LR-scheduler flavor: `cosine_anealing_lr.sh`, `cosine_anealing_warm_lr.sh`, `cyclic_lr.sh`, `expr_lr.sh`, `fixed_lr.sh`), writing them to the desired location.
- `train_demnet_FL/` and `train_demnet_FL_V2/` — launch Federated-Learning training of the DemNet model. The V2 subfolder is a **second iteration**: in V1, train/validation split and client partitioning happened at Flower server instantiation time (inside `prepare_data_for_FL_training` in `server_app.py`); V2 instead consumes pre-computed index files produced by `scripts_python/data_manipulation/create_idx_files_for_federated_simulations.py`, launched by the shell script before the Flower app itself.
- `train_demnet_centralized/` and `train_demnet_centralized_V2/` — centralized (non-federated) counterparts, one script per namesake in the FL folder. In V2, most hyperparameters are *not* set directly in these scripts — they're read from the config files produced by the FL V2 scripts (per its README, this path was "not tested since `output_channels_first_layer` was added to the model config").
- `train_demnet_centralized_V2_IND` — a further centralized V2 variant (naming suggests "independent" runs; not separately documented with its own README at time of writing — check the scripts themselves if you need the exact distinction from `train_demnet_centralized_V2`).
- `train_vgg_centralized/`, `train_vgg_FL/`, `train_vgg_FL_V2/` — VGG equivalents of the demnet training scripts above (centralized and federated).
- `train_scripts_submission/` — likely scripts used to batch-submit the above training jobs to SLURM; inspect contents directly if you need submission-queue details.

## Notes for future work

- These scripts were explicitly written for one-off runs/experiments, not shipped as part of the package — expect hardcoded paths, author-specific HPC partition/account names, and parameters tuned for specific experiments.
- When adding a new model or training variant, the existing `_FL_V2` / `_centralized_V2` pairing (shared config files, FL writes them, centralized reads them) is the pattern to follow if you want parity with the latest approach in this folder — though `scripts_sh_V2/` (see `scripts_sh_V2.md`) is the more current generation overall.
