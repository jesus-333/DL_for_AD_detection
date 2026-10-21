# `scripts_sh_V2/` (current shell-script generation)

**Path:** `scripts_sh_V2/`

## What it is

The **newer, improved** generation of shell launch scripts (superseding most of `scripts_sh/`, see `scripts_sh.md`), targeting centralized training of the newer model architectures (ViT, Swin, VAV, DemNet) on the HPC SLURM cluster. No top-level README exists in this folder at time of writing — the summary below is derived directly from the scripts and their comments.

## Layout

- `0_launch_sbatch_cent.sh` — main SBATCH entry point for a centralized training run: picks a model (`model_name`, e.g. `"vav"`), an optimizer (`AdamW` or `SGD`) and an LR scheduler (`ExponentialLR` or `CosineAnnealingWarmRestarts`), generates a random seed, and submits the job. Contains inline notes/gotchas as comments, e.g. "for demnet you have to set `use_rgb_input` to false inside `dataset.toml`" and a pointer to a note in `demnet.sh` about `NUM_CLASSES`/`MERGE_AD_CLASS`.
- `0_launch_sbatch_cent_continuation.sh` — like the above, but for resuming/continuing a previously started run (checkpoint continuation) rather than starting fresh.
- `0_launch_debug.sh` / `0_launch_debug_continuation.sh` — debug-mode counterparts of the two scripts above, for local/interactive iteration rather than full SBATCH submission.
- `0_decompress.sh` — a one-off interactive-mode utility to decompress downloaded/preprocessed ADNI MRI zip archives (`Brain_MRI_Preprocess__GradWarp-B1-N3__-_SCALED_MRI_*.zip`) from `$SCRATCH/ADNI_MRI_NORMALIZED/` into `$SCRATCH/ADNI_MRI_NORMALIZED_DECOMPRESSED/`, using `7z`.
- `demnet_cent.sh`, `vit_cent.sh`, `swin_cent.sh`, `vav_cent.sh` — per-model SLURM job scripts (SBATCH directives: GPU partition `hopper`, qos `iris-hopper`, etc.) that actually run centralized training for DemNet / ViT / Swin / VAV respectively; these are what `0_launch_sbatch_cent*.sh` submits.
- `eternal_grokking.sh` — currently just a shebang line (`#!/bin/sh`), effectively empty. Untracked/in-progress at time of writing — likely scaffolding for a long-running "grokking" experiment (see `config/seed_list_grokking/` below, which has `demnet.txt` / `vit.txt` seed lists already prepared for it).
- `config/` — TOML configuration files consumed by the scripts above:
  - Top-level reusable configs: `dataset.toml`, `server.toml`, `training.toml`, `optimizer_AdamW.toml` / `optimizer_AdamW_backup.toml` / `optimizer_SGD.toml`, `lr_sched_ExponentialLR.toml` / `lr_sched_CosineAnnealingWarmRestarts.toml`, one `model_*.toml` per architecture (`model_demnet.toml`, `model_demnet_tiny.toml`, `model_swin.toml`, `model_vav.toml`, `model_vgg.toml`, `model_vit.toml`, `model_vit_micro.toml`, `model_vit_tiny.toml`), plus `chained_lr_scheduler_example.toml`.
  - `config/debug/train_<model>_<random_id>/` — a large number of **auto-generated, per-run** config snapshots (`dataset.toml`, `lr_scheduler.toml`, `model.toml`, `optimizer.toml`, `training.toml`), one directory per debug run, named by a random numeric id. These are run artifacts, not hand-maintained config — do not treat them as a place to edit defaults.
  - `config/seed_list_grokking/` — seed lists (`demnet.txt`, `vit.txt`) for the in-progress grokking experiment mentioned above.
  - `config/training_hpc/` — present but essentially empty at time of writing (only a `.DS_Store`).

## Notes for future work

- This is the folder to extend when adding a new model/training variant going forward, following the `0_launch_sbatch_cent.sh` → per-model `*_cent.sh` → `config/model_*.toml` pattern already established for demnet/vit/swin/vav.
- `config/debug/train_*_<id>/` directories are generated, numerous, and disposable — don't document or rely on specific ones; they're a side effect of running debug launches, not source of truth.
- `eternal_grokking.sh` and `config/seed_list_grokking/` look like an in-progress/unfinished initiative — confirm with the user before building on top of it or assuming its intended behavior.
