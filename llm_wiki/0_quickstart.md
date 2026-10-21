# Quickstart

This is an internal **LLM/agent-facing wiki**: documentation written as a support tool for Claude (and other coding agents) doing future development on this codebase, not an end-user guide. Keep it in sync as the code changes, and use it as the first stop before re-deriving architecture from scratch by reading files cold.

## What this repository is

`DL_for_AD_detection` (package name `addl`, project name `Alzheimer_Deep_Learning_Detection` in `pyproject.toml`) applies Deep Learning — including Federated Learning — to Alzheimer's Disease (AD) detection/diagnosis. The work is currently scoped to **brain MRI data** (primarily the [ADNI dataset](https://adni.loni.usc.edu/data-samples/adni-data/)), and is expected to expand over time to other imaging and non-imaging data modalities. The project is still at an early stage (per the top-level `README.md`).

## Repository layout

- **`src/addl/`** — the installable Python package (built via [hatchling](https://hatch.pypa.io/latest/); `pyproject.toml` maps `src/addl` → importable `addl`). This is the only part of the repo meant to be "the product" — everything else is scripts and experiments that use it.
  - `dataset/` — dataset classes and support code for loading MRI data (HDF5, PNG, raw formats), ADNI- and Kaggle-specific helpers.
  - `federated/` — Federated-Learning client/server code built on [Flower](https://flower.ai/).
  - `model/` — model architectures: DemNet (custom CNN), VGG, Swin Transformer, ViT, and VAV (Volume-Attention ViT).
  - `training/` — shared training-loop machinery (used by both centralized and federated training), metrics, checkpointing.
  - `other/` — Weights & Biases (wandb) logging integration.
  - `cli_scripts/` + `cli.py` — the implementations and argparse wiring behind the package's installed CLI commands (`nii-to-hdf5`, `pt-to-npy`, `npy-to-pt`, `get-hdf5-dataset-length`, `wandb-get-run-status`, `wandb-download-project-runs-metrics`; see `pyproject.toml`'s `[project.scripts]`).
  - Detailed, per-file documentation for all of `src/addl/` lives under [`llm_wiki/addl/`](addl/), mirroring the package's own folder structure — one `.md` per `.py` file.
- **`scripts_sh/`**, **`scripts_sh_V2/`**, **`scripts_python/`** — standalone scripts (shell + Python) for training, data preprocessing/conversion, and analysis. These were **never meant to become part of the installable package** — they're one-off / experiment-specific code the author wrote for training runs, data munging, and analysis, so expect superseded versions (`_V1`, `_V2`, ...), hardcoded paths, and HPC-specific (SLURM) assumptions. Brief, folder-level summaries live under [`llm_wiki/scripts/`](scripts/):
  - [`scripts_sh.md`](scripts/scripts_sh.md) — original shell launch scripts (mostly SLURM/HPC).
  - [`scripts_sh_V2.md`](scripts/scripts_sh_V2.md) — current-generation shell launch scripts.
  - [`scripts_python.md`](scripts/scripts_python.md) — Python scripts for training, analysis, data manipulation, plotting, and scratch/test code.
- **`config/`**, **`config_example/`** — TOML configuration files/templates used during training.
- **`data/`**, **`model_weights/`**, **`Saved Model/`** — data and trained-model artifacts (not source code).

## Coding style

All code in this repository (package and scripts alike) should follow the author's coding-style preferences, copied into this wiki at [`coding_style_instructions_ENG.md`](coding_style_instructions_ENG.md). Highlights: snake_case everywhere (including classes), spaces around `=` and before `:` in type annotations/`if` conditions, aligned `=` signs in multi-line calls/dict literals, numpydoc-style docstrings, f-strings for interpolation, hyphen-separated multi-word CLI command names, and section dividers (`# - - - ... - - -` between top-level blocks, `# %%%...%%%` inside functions). **Read that file before writing or editing any code here.**

## Where to go next

- Need to understand one specific file in the package? Go straight to its doc under [`llm_wiki/addl/`](addl/) (same relative path, `.md` instead of `.py`).
- Need the full list of everything in this wiki? See [`0_index.md`](0_index.md).
- Need "why was it built this way" / historical context rather than current-code description? Check this repo's Hindsight knowledge pages (`hindsight_search_knowledge_pages`) — they track decisions and initiatives over time; this wiki tracks what the code currently *is*.
