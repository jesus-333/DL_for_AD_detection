# Index

Full index of this wiki. Start at [`0_quickstart.md`](0_quickstart.md) if you haven't already.

## Top level

- [`0_quickstart.md`](0_quickstart.md) — project overview, repo layout, where to go next.
- [`0_index.md`](0_index.md) — this file.
- [`coding_style_instructions_ENG.md`](coding_style_instructions_ENG.md) — author's coding-style guidelines; apply to all code in this repo.

## `addl/` — one `.md` per `.py` file in `src/addl/`, same relative path

- [`addl/__init__.md`](addl/__init__.md) — `src/addl/__init__.py`
- [`addl/cli.md`](addl/cli.md) — `src/addl/cli.py`

### `addl/cli_scripts/`
- [`addl/cli_scripts/__init__.md`](addl/cli_scripts/__init__.md) — `src/addl/cli_scripts/__init__.py`
- [`addl/cli_scripts/nii_to_hdf5.md`](addl/cli_scripts/nii_to_hdf5.md) — `src/addl/cli_scripts/nii_to_hdf5.py`
- [`addl/cli_scripts/pt_to_npy.md`](addl/cli_scripts/pt_to_npy.md) — `src/addl/cli_scripts/pt_to_npy.py`
- [`addl/cli_scripts/stats.md`](addl/cli_scripts/stats.md) — `src/addl/cli_scripts/stats.py`
- [`addl/cli_scripts/wandb_scripts.md`](addl/cli_scripts/wandb_scripts.md) — `src/addl/cli_scripts/wandb_scripts.py`

### `addl/dataset/`
- [`addl/dataset/dataset.md`](addl/dataset/dataset.md) — `src/addl/dataset/dataset.py`
- [`addl/dataset/dataset_HDF5.md`](addl/dataset/dataset_HDF5.md) — `src/addl/dataset/dataset_HDF5.py`
- [`addl/dataset/dataset_png.md`](addl/dataset/dataset_png.md) — `src/addl/dataset/dataset_png.py`
- [`addl/dataset/support_dataset.md`](addl/dataset/support_dataset.md) — `src/addl/dataset/support_dataset.py`
- [`addl/dataset/support_dataset_ADNI.md`](addl/dataset/support_dataset_ADNI.md) — `src/addl/dataset/support_dataset_ADNI.py`
- [`addl/dataset/support_dataset_kaggle.md`](addl/dataset/support_dataset_kaggle.md) — `src/addl/dataset/support_dataset_kaggle.py`

### `addl/federated/`
- [`addl/federated/__init__.md`](addl/federated/__init__.md) — `src/addl/federated/__init__.py`
- [`addl/federated/client.md`](addl/federated/client.md) — `src/addl/federated/client.py`
- [`addl/federated/server.md`](addl/federated/server.md) — `src/addl/federated/server.py`
- [`addl/federated/server_backup.md`](addl/federated/server_backup.md) — `src/addl/federated/server_backup.py`
- [`addl/federated/support_federated_generic.md`](addl/federated/support_federated_generic.md) — `src/addl/federated/support_federated_generic.py`
- [`addl/federated/support_federated_server.md`](addl/federated/support_federated_server.md) — `src/addl/federated/support_federated_server.py`

### `addl/model/`
- [`addl/model/__init__.md`](addl/model/__init__.md) — `src/addl/model/__init__.py`
- [`addl/model/demnet.md`](addl/model/demnet.md) — `src/addl/model/demnet.py`
- [`addl/model/download_published_model.md`](addl/model/download_published_model.md) — `src/addl/model/download_published_model.py`
- [`addl/model/support_model.md`](addl/model/support_model.md) — `src/addl/model/support_model.py`
- [`addl/model/swin.md`](addl/model/swin.md) — `src/addl/model/swin.py`
- [`addl/model/vav.md`](addl/model/vav.md) — `src/addl/model/vav.py`
- [`addl/model/vgg_nets.md`](addl/model/vgg_nets.md) — `src/addl/model/vgg_nets.py`
- [`addl/model/vit.md`](addl/model/vit.md) — `src/addl/model/vit.py`

### `addl/other/`
- [`addl/other/wandb_interactions.md`](addl/other/wandb_interactions.md) — `src/addl/other/wandb_interactions.py`

### `addl/training/`
- [`addl/training/__init__.md`](addl/training/__init__.md) — `src/addl/training/__init__.py`
- [`addl/training/metrics.md`](addl/training/metrics.md) — `src/addl/training/metrics.py`
- [`addl/training/support_training.md`](addl/training/support_training.md) — `src/addl/training/support_training.py`
- [`addl/training/test_functions.md`](addl/training/test_functions.md) — `src/addl/training/test_functions.py`
- [`addl/training/train_functions.md`](addl/training/train_functions.md) — `src/addl/training/train_functions.py`

## `scripts/` — brief, folder-level summaries of the non-package script trees

- [`scripts/scripts_sh.md`](scripts/scripts_sh.md) — `scripts_sh/` (original shell launch scripts)
- [`scripts/scripts_sh_V2.md`](scripts/scripts_sh_V2.md) — `scripts_sh_V2/` (current shell launch scripts)
- [`scripts/scripts_python.md`](scripts/scripts_python.md) — `scripts_python/` (training/analysis/data-manipulation/plot/test Python scripts)
