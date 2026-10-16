# support_training.py

**Path:** `src/addl/training/support_training.py`

## Purpose
Grab-bag of training infrastructure used by `train_functions.py` (and indirectly by the federated code in `addl.federated`, which builds the same `training_config`/`optimizer_config`/`lr_scheduler_config` dicts): config validation/defaulting, dataloader construction, optimizer/LR-scheduler factories, epoch-level checkpoint backup, wandb log-dict bookkeeping, and CPU/GPU memory profiling helpers. This is the file that defines **what keys a training config dict must/can contain** — it is the closest thing this package has to a schema for `training.toml` / `optimizer.toml` / `lr_scheduler.toml`.

The overall pattern in this file: every `check_*_config(config)` function mutates `config` **in place**, filling in missing optional keys with defaults (printing a `Warning:` to stdout when it does) and raising `ValueError` for missing required keys. Callers must call the relevant `check_*_config` before trusting the dict has fully-populated keys.

## Public API

### `backup_every_epoch(path_to_save: str, epoch: int, model, optimizer=None, lr_scheduler=None)`
Writes `model_backup_A.pth` on even epochs and `model_backup_B.pth` on odd epochs (and likewise `optimizer_backup_{A,B}.pth`, `lr_scheduler_backup_{A,B}.pth` if those args are given) into `path_to_save`, **overwriting the previous same-parity file each time**. Purpose: keep a "backup of the backup" so that if an HPC job gets killed mid-`torch.save` (observed in practice — corrupting the file being written), the other-parity file from one epoch earlier is still intact and usable to resume training. This is distinct from the epoch-numbered checkpoints (`model_{epoch}.pth`) saved by `train_functions.train()`.

### `update_log_dict_metrics(metrics_dict: dict, log_dict: dict, label: str | None = None)`
Merges `metrics_dict` into `log_dict` (used to build the per-epoch wandb log payload), prefixing each key with `f'{key}_{label}'` if `label` is given, else leaving keys unprefixed. E.g. `{'accuracy': 0.9}` + `label='train'` → `log_dict['accuracy_train'] = 0.9`.

### `check_training_config(config: dict)`
Validates/defaults the top-level training config. Required keys (raise `ValueError` if missing): `batch_size`, `epochs` (must be `> 0`), `optimizer_config` (further validated via `check_optimizer_config`). Optional keys defaulted if missing: `dataloader_config` (→ `{}`, then validated via `check_dataloader_config`), `use_scheduler` (→ `False`), `device` (→ `"cpu"`), `path_to_save_model` (→ `"model_weights"`), `epoch_to_save_model` (→ `epochs + 2`, i.e. effectively "never, save only at the end" — same fallback if given a non-positive value), `measure_metrics_during_training` (→ `True`), `print_var` (→ `True`), `seed` (→ random `np.random.randint(0, 2**32 - 1)` if missing or negative). If `use_scheduler` is `True`, `lr_scheduler_config` becomes required and is validated via `check_lr_scheduler_config`.
- **Wandb block:** if `wandb_training` is `True` (defaults to `False` if missing), then `project_name` and `model_artifact_name` become required (raise if missing); `log_freq` defaults to `1`; `name_training_run` defaults to `None`; `debug` defaults to `False` ("not used in the training functions... only useful if you want to quickly filter the training run in wandb").
- **FL-only block:** if the key `fl_training` is present in config (any value — just a presence check, not a boolean check), `use_weights_with_lower_validation_error` defaults to `False` if missing.
- **VGG-only block:** if the key `vgg_training` is present, `use_vgg_normalization_values` defaults to `True` if missing.
- Note: `check_training_config` does **not** itself validate `lr` (only `check_optimizer_config` does, as `optimizer_config['lr']`) — there is no top-level `lr` key despite `train_functions.train`'s docstring listing one (docstring appears stale — see `train_functions.md`).

### `get_dataloader(dataset, batch_size: int, dataloader_config: dict = None) -> torch.utils.data.DataLoader`
Builds a `DataLoader` with `shuffle=True` always, plus `num_workers`, `prefetch_factor`, `pin_memory`, `persistent_workers` taken from `dataloader_config` (validated/defaulted via `check_dataloader_config` if `dataloader_config` is `None` or incomplete).

### `check_dataloader_config(dataloader_config: dict)`
Defaults: `num_workers` → `0`; `prefetch_factor` → `2` if `num_workers > 0` else `None` (and forces `None` if `num_workers == 0` but a non-`None` value was explicitly given, to avoid a PyTorch error); `pin_memory` → `False`; `persistent_workers` → `False` (forced to `False` if `num_workers == 0`, again to avoid a PyTorch error).

### `get_optimizer(optimizer_config: dict, model: torch.nn.Module) -> torch.optim.Optimizer`
Factory keyed on `optimizer_config['name']`. Implemented: `'Adam'`, `'AdamW'`, `'LBFGS'`, `'SGD'`. Raises `ValueError` for any other name. Each branch passes a fixed, hand-picked subset of `torch.optim.<Name>`'s constructor args read directly from `optimizer_config` (e.g. `AdamW` reads `lr`, `betas`, `eps`, `weight_decay`, `amsgrad`, `maximize` — but NOT `capturable`/`differentiable`/`foreach`, which are deliberately left at PyTorch's defaults and not configurable here).

### `check_optimizer_config(optimizer_config: dict)`
Requires `lr` (raises if missing or `<= 0`). Then, branching on `optimizer_config['name']` (itself assumed already present — not checked here, only inside `get_optimizer`/`get_lr_scheduler`-style dispatch, so a missing `'name'` key will raise a `KeyError`, not a friendly `ValueError`, the first time this function branches on it), defaults are filled per-optimizer:
- `Adam`: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0`, `decoupled_weight_decay=False`, `amsgrad=False`, `maximize=False`.
- `AdamW`: same as Adam but `weight_decay=1e-2` default and no `decoupled_weight_decay` key (AdamW is always decoupled).
- `LBFGS`: `max_iter=20`, `max_eval=max_iter*1.25`, `tolerance_grad=1e-7`, `tolerance_change=1e-9`, `history_size=100`, `line_search_fn=None`.
- `SGD`: `momentum=0`, `dampening=0`, `weight_decay=0`, `nesterov=False`, `maximize=False`.
- Unknown name → `ValueError`.

### `get_lr_scheduler(lr_scheduler_config: dict, optimizer) -> torch.optim.lr_scheduler._LRScheduler`
Factory keyed on `lr_scheduler_config['name']`. Implemented: `'ExponentialLR'`, `'CosineAnnealingLR'`, `'CosineAnnealingWarmRestarts'`, `'StepLR'`, `'CyclicLR'`, and a composite **`'ChainedScheduler'`**: its config must supply `list_config_schedulers`, a dict of named sub-configs, each itself a valid scheduler config (recursively built via a recursive call to `get_lr_scheduler`), combined into `torch.optim.lr_scheduler.ChainedScheduler`. See `scripts_sh_V2/config/chained_lr_scheduler_example.toml` for an example on-disk config. Unknown name → `ValueError`.

### `check_lr_scheduler_config(lr_scheduler_config: dict)`
Requires `name`. Validates/defaults per scheduler type analogous to `check_optimizer_config` (e.g. `ExponentialLR` requires `gamma`; `CosineAnnealingWarmRestarts` requires `T_0`, defaults `T_mult=1`, `eta_min=0`; `CyclicLR` requires `base_lr`, `max_lr`, `step_size_up`, `step_size_down`, defaults `mode='triangular'`, `gamma=1`). For `'ChainedScheduler'`, recursively calls itself on every entry of `list_config_schedulers`. Unknown name → `ValueError`.

### `cpu_memory_usage_in_gb(func, *args, **kwargs) -> float`
Runs `func(*args, **kwargs)` on the current thread while a background `Thread` polls `psutil.Process().memory_info().rss` every 0.1s; returns peak RSS minus the baseline RSS measured just before starting, in GB. Explicitly noted as copied from an external blog post (analyticsvidhya.com).

### `gpu_start_memory_tracking()` / `gpu_print_memory_usage()` / `gpu_cleanup()`
Thin wrappers around `torch.cuda.reset_peak_memory_stats()` / `torch.cuda.max_memory_allocated()` / `gc.collect()` + `torch.cuda.empty_cache()` for GPU memory profiling during debugging. Also copied from the same external source. `gpu_cleanup` sleeps 3 seconds to "allow time for memory to clear" before reporting.

## Notes
- This file is the de-facto schema reference for `training.toml`, `optimizer_*.toml`, `lr_scheduler_*.toml` found under `scripts_sh_V2/config/` and `config/`/`config_example/` — reading the `check_*_config` functions is the fastest way to know every accepted key and its default/required status for a given optimizer/scheduler/dataloader choice.
- Defaults are applied by mutating the passed-in dict; callers rely on this (e.g. `train_functions.train()` reads `training_config['dataloader_config']` right after calling `check_training_config`, trusting it now exists).
- `check_optimizer_config`/`get_optimizer` and `check_lr_scheduler_config`/`get_lr_scheduler` both branch on `config['name']` without checking its presence first in `check_*` — a config missing `'name'` raises a raw `KeyError` rather than the descriptive `ValueError`s used elsewhere.
- The memory-profiling helpers (`cpu_memory_usage_in_gb`, `gpu_*`) are standalone debug utilities, not wired into the main `train`/`test` flow — they're meant to be called ad hoc (e.g. from `scripts_python/test_code/check_memory_usage_load_data.py` or `scripts_sh/debug/check_memory_usage.sh`).
