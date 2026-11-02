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
4. **(optional) Resume / continuation.** If `training_config['resume_from_backup']` is `True`, calls `support_training.load_latest_checkpoint(path_to_save_model)`, which returns the newest valid `checkpoint_backup_{A,B}.pth` (or `None`). If a checkpoint is found it is used for model, optimizer, LR scheduler, epoch and best validation loss, and `path_past_weights`/`path_past_optimizer` are **ignored**. Otherwise, if `training_config.get('path_past_weights')` is not `None`, a prior model `state_dict` is loaded (first launch of a continuation of an old run). Note that `None` values are not written to toml, so usually the key is simply absent.
5. Always uses `torch.nn.CrossEntropyLoss()` (hardcoded — "TODO Add option for other loss function").
6. Builds the optimizer via `support_training.get_optimizer`, then loads its state from the checkpoint or, if there is no checkpoint, from `path_past_optimizer` (if present).
7. **(optional)** Builds an LR scheduler via `support_training.get_lr_scheduler` if `training_config['use_scheduler']`; its state is restored from the checkpoint if present.
8. Creates `path_to_save_model` and `path_to_save_snapshot` if missing.
9. **(optional)** If wandb training and `wandb_watch_log != 'none'`, calls `wandb.watch(model, log = wandb_watch_log, log_freq = wandb_watch_log_freq, log_graph = wandb_watch_log_graph)`. Defaults reproduce the old behaviour (`'all'`, `log_freq`, `True`); with `log_freq = 1` this logs histograms of every parameter/gradient every batch and was the main cause of local wandb folders filling the disk quota in long trainings.
10. Sets `start_epoch` = checkpoint epoch + 1 (and `best_loss_val` from the checkpoint), or `training_config['start_epoch']` (default 0) if there is no checkpoint. **The epoch counter is global**: the loop is `range(start_epoch, epochs)`, so for a resumed run `epochs` is the total target, not the number of additional epochs.
11. Per-epoch loop, in this order (the order matters for resuming):
   - `train_epoch_function(...)` — one backward-pass epoch over `train_loader`.
   - **(optional)** If `validation_dataset` was given: `validation_epoch_function(...)`; if this epoch's validation loss is the best so far, overwrites `model_BEST.pth` / `optimizer_BEST.pth` (atomic save).
   - **(optional)** If `training_config['measure_metrics_during_training']`: computes train (and validation) metrics via `metrics.compute_metrics` and appends them into `computed_metrics_during_training` (keys like `accuracy_train`, `accuracy_validation`, etc.).
   - **(optional)** Steps the LR scheduler (`lr_scheduler.step()`) if one was configured.
   - Saves `model_{epoch+1}.pth` (+ `optimizer_{epoch+1}.pth`) in `path_to_save_snapshot` every `epoch_to_save_model` epochs.
   - **(optional)** If `backup_model_every_epoch`, calls `support_training.backup_every_epoch(...)` (alternating A/B files). If `resume_from_backup` is also `True`, passes `training_state = {'epoch', 'best_loss_val'}` so a full `checkpoint_backup_{A,B}.pth` is written too. Backups are written **after** validation and the scheduler step, so the saved best loss and optimizer/scheduler state are exactly what the next epoch needs.
   - **(optional)** Prints epoch summary (`print_var`), including a `model.print_debug()` call if `training_config['debug']` is set **and** the model exposes a callable `print_debug` attribute.
   - **(optional)** Logs the epoch's `log_dict` to wandb (`wandb.log(log_dict)`), including `epoch` and, on snapshot epochs with `log_model_artifact`, the snapshot file. `validation_loss` is logged only if there is a validation set.
   - **(optional)** If `stop_file_path` is set and the file exists, breaks out of the loop (after backup and log of the current epoch).
12. After the loop: always saves a final `model_END.pth` (also when stopped through the stop file). If wandb training and `log_model_artifact`, logs `model_END.pth` (and `model_BEST.pth` if validation was used) to the wandb artifact/`wandb.save`.
13. Returns `(model, computed_metrics_during_training)`.

### `wandb_train(config: dict, model, train_dataset, validation_dataset=None) -> (model, computed_metrics_during_training)`
The wandb-aware entry point — use this instead of `train()` directly when you want run tracking. Requires `config` to have both `training_config` and `model_config` keys (the latter is only used for logging to wandb, not consumed by training logic). Opens a `wandb.init(...)` context (`project=training_config['project_name']`, `id=str(training_config['seed'])`, `resume='allow'` — meaning **the wandb run id is derived from the seed**, so re-running with the same seed resumes/overwrites the same wandb run rather than creating a new one), if `training_config['resume_from_backup']` is set, calls `run.define_metric('*', step_metric = 'epoch')` so every metric uses the logged epoch as x-axis (the wandb step keeps increasing across resumes and is not equal to the epoch), optionally creates a `wandb.Artifact` for the trained model (if `training_config['log_model_artifact']`), calls `train(...)` internally, then logs the artifact with `run.log_artifact(...)`.

### `train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict=None, print_var=True) -> train_loss: float`
One standard supervised-learning epoch: `model.train()`, iterate batches, zero grad, forward (`model(x)` — note: **not** `model.classify(x)`, unlike `metrics.compute_label`), compute loss, backward, optimizer step, accumulate loss weighted by batch size, return the dataset-averaged loss (`train_loss / len(train_loader.sampler)`). Writes `log_dict['train_loss']` if `log_dict` is given.

### `validation_epoch_function(model, validation_dataloader, loss_function, device, log_dict=None) -> validation_loss: float`
Same as above but `model.eval()`, wrapped in `torch.no_grad()`, no backward/optimizer step. Writes `log_dict['validation_loss']` if given — this is the key that `test_functions.test()` relies on and re-labels.

## Notes
- **`train`'s docstring is stale/incomplete**: it lists `lr` as a required top-level `training_config` key, but `support_training.check_training_config` never checks for a top-level `lr` (only `optimizer_config['lr']`). Trust the code in `support_training.py` over this docstring for the actual accepted/required keys.
- **Dead/no-op branching:** the end-of-training save block (`if training_config['backup_model_every_epoch']: ... else: ...`) computes the exact same `model_file_path_END` string in both branches — the `if` currently has no effect. There's a `# TODO Implement the removal of the backup file created during the training?` comment noting this is unfinished.
- **Unguarded config key read without default:** `training_config['log_model_artifact']` is read directly in `train()` but is **not** given a default by `support_training.check_training_config` — callers must supply it when `wandb_training=True`. (`backup_model_every_epoch` now defaults to `False`.)
- `train_epoch_function`/`validation_epoch_function` use `model(x)` directly for the forward pass (raw logits in, `CrossEntropyLoss` expects logits), whereas `metrics.compute_label` uses `model.classify(x)` — every model in `addl.model` must implement both a standard `forward`/`__call__` and a separate `.classify()` that returns hard predicted labels.
- **Resumable ("eternal") trainings:** all the resume keys (`resume_from_backup`, `start_epoch`, `path_to_save_snapshot`, `stop_file_path`, `wandb_watch_*`) are opt-in with defaults that keep the old behaviour. This matters because `addl.federated.client` calls `train()` every FL round with the same config. Launcher: `scripts_sh_V2/eternal_grokking.sh` (see `scripts/scripts_sh_V2.md`). RNG state (data shuffling) is **not** saved, so a resumed run is not bit-identical to an uninterrupted one.
- wandb's run `id` is tied to the **seed** (deliberately: the same code either creates a new run or continues an existing one, e.g. for continuations and requeued jobs), not to a unique identifier per invocation — be aware this means two different training attempts sharing a seed value will resume/collide on the same wandb run.
- All checkpoint paths are built with manual `'{}/{}'.format(...)` / f-string path joins rather than `os.path.join`, except where `os.path.join` is used in `support_training.backup_every_epoch`.
