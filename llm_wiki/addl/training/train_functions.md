# train_functions.py

**Path:** `src/addl/training/train_functions.py`

## Purpose
The core centralized training loop for this package. This is the function both direct centralized-training scripts (`scripts_python/training/*.py`) and (indirectly, via config dicts built elsewhere) the federated client code in `addl.federated` ultimately call to actually fit a model for some number of local epochs. It owns: dataloader setup, optimizer/scheduler construction (delegating to `support_training`), the per-epoch train/validate cycle, metric computation (delegating to `metrics`), checkpointing (periodic, best-validation, backup-every-epoch, end-of-training), and optional Weights & Biases (wandb) run tracking/model-artifact logging.

## Public API

### `train(training_config: dict, model, train_dataset, validation_dataset=None, wandb_model_artifact=None) -> (model, computed_metrics_during_training: dict)`
The main training loop, **without** wandb run management (no `wandb.init`) — assumes if `training_config['wandb_training']` is `True`, a wandb run is already active and `wandb_model_artifact` was passed in (normally by `wandb_train`, see below — calling `train()` directly with `wandb_training=True` but no active run will fail at `wandb.log`/`wandb.watch`).

Flow, in order:
1. `support_training.check_training_config(training_config)` — validates/defaults the config in place (see `support_training.md` for the full key reference).
2. Builds `train_loader` (and `validation_loader` if `validation_dataset is not None`) via `support_training.get_dataloader`.
3. Moves `model` to `training_config['device']`.
4. **(optional)** If `training_config['path_past_weights']` is present, loads a prior `state_dict` into `model` — used to resume/continue training.
5. Always uses `torch.nn.CrossEntropyLoss()` (hardcoded — "TODO Add option for other loss function").
6. Builds the optimizer via `support_training.get_optimizer`.
7. **(optional)** If `training_config['path_past_optimizer']` is present, loads a prior optimizer `state_dict` — pairs with `path_past_weights` for resuming.
8. **(optional)** Builds an LR scheduler via `support_training.get_lr_scheduler` if `training_config['use_scheduler']`.
9. Creates `training_config['path_to_save_model']` if missing.
10. **(optional)** If wandb training, calls `wandb.watch(model, ...)` to track gradients/parameters.
11. Per-epoch loop (`training_config['epochs']` iterations):
   - `train_epoch_function(...)` — one backward-pass epoch over `train_loader`.
   - Saves a numbered checkpoint `model_{epoch+1}.pth` (+ `optimizer_{epoch+1}.pth`) every `training_config['epoch_to_save_model']` epochs.
   - **(optional)** If `training_config['backup_model_every_epoch']` is truthy, calls `support_training.backup_every_epoch(...)` (alternating A/B backup files) — note this key is read unconditionally (no default applied anywhere in `check_training_config`), so it must be present in the config or this will raise `KeyError`.
   - **(optional)** If `validation_dataset` was given: `validation_epoch_function(...)`; if this epoch's validation loss is the best so far, overwrites `model_BEST.pth` / `optimizer_BEST.pth`.
   - **(optional)** If `training_config['measure_metrics_during_training']`: computes train (and validation) metrics via `metrics.compute_metrics` and accumulates them epoch-by-epoch into `computed_metrics_during_training` (keys like `accuracy_train`, `accuracy_validation`, etc., built from whatever keys `metrics.compute_metrics_from_labels` returns).
   - **(optional)** Steps the LR scheduler (`lr_scheduler.step()`) if one was configured.
   - **(optional)** Prints epoch summary (`print_var`), including a `model.print_debug()` call if `training_config['debug']` is set **and** the model exposes a callable `print_debug` attribute (checked defensively via `getattr`).
   - **(optional)** Logs the epoch's `log_dict` to wandb (`wandb.log(log_dict)`), including adding the saved checkpoint file to `wandb_model_artifact` on checkpoint epochs if `training_config['log_model_artifact']` is set.
12. After the loop: always saves a final `model_END.pth` (the `if/else` branches for this are currently identical dead code — see Notes). If wandb training and `log_model_artifact`, logs `model_END.pth` (and `model_BEST.pth` if validation was used) to the wandb artifact/`wandb.save`.
13. Returns `(model, computed_metrics_during_training)`.

### `wandb_train(config: dict, model, train_dataset, validation_dataset=None) -> (model, computed_metrics_during_training)`
The wandb-aware entry point — use this instead of `train()` directly when you want run tracking. Requires `config` to have both `training_config` and `model_config` keys (the latter is only used for logging to wandb, not consumed by training logic). Opens a `wandb.init(...)` context (`project=training_config['project_name']`, `id=str(training_config['seed'])`, `resume='allow'` — meaning **the wandb run id is derived from the seed**, so re-running with the same seed resumes/overwrites the same wandb run rather than creating a new one), optionally creates a `wandb.Artifact` for the trained model (if `training_config['log_model_artifact']`), calls `train(...)` internally, then logs the artifact with `run.log_artifact(...)`.

### `train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict=None, print_var=True) -> train_loss: float`
One standard supervised-learning epoch: `model.train()`, iterate batches, zero grad, forward (`model(x)` — note: **not** `model.classify(x)`, unlike `metrics.compute_label`), compute loss, backward, optimizer step, accumulate loss weighted by batch size, return the dataset-averaged loss (`train_loss / len(train_loader.sampler)`). Writes `log_dict['train_loss']` if `log_dict` is given.

### `validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict=None) -> validation_loss: float`
Same as above but `model.eval()`, wrapped in `torch.no_grad()`, no backward/optimizer step. Writes `log_dict['validation_loss']` if given — this is the key that `test_functions.test()` relies on and re-labels.

## Notes
- **`train`'s docstring is stale/incomplete**: it lists `lr` as a required top-level `training_config` key, but `support_training.check_training_config` never checks for a top-level `lr` (only `optimizer_config['lr']`). Trust the code in `support_training.py` over this docstring for the actual accepted/required keys.
- **Dead/no-op branching:** the end-of-training save block (`if training_config['backup_model_every_epoch']: ... else: ...`) computes the exact same `model_file_path_END` string in both branches — the `if` currently has no effect. There's a `# TODO Implement the removal of the backup file created during the training?` comment noting this is unfinished.
- **Unguarded config keys read without defaults:** `training_config['backup_model_every_epoch']` and `training_config['log_model_artifact']` are read directly in `train()` but are **not** given defaults by `support_training.check_training_config` — callers must supply them explicitly (`log_model_artifact` only matters when `wandb_training=True`; `backup_model_every_epoch` is read unconditionally every epoch regardless of wandb).
- `train_epoch_function`/`validation_epoch_function` use `model(x)` directly for the forward pass (raw logits in, `CrossEntropyLoss` expects logits), whereas `metrics.compute_label` uses `model.classify(x)` — every model in `addl.model` must implement both a standard `forward`/`__call__` and a separate `.classify()` that returns hard predicted labels.
- wandb's run `id` is tied to the **seed**, not to a unique identifier per invocation — be aware this means two different training attempts sharing a seed value will resume/collide on the same wandb run.
- All checkpoint paths are built with manual `'{}/{}'.format(...)` / f-string path joins rather than `os.path.join`, except where `os.path.join` is used in `support_training.backup_every_epoch`.
