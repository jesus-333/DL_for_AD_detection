# test_functions.py

**Path:** `src/addl/training/test_functions.py`

## Purpose
Marked `TODO : Implement in the future` in the module docstring — this file is a thin, not-fully-finished wrapper for evaluating an already-trained model against a held-out dataset (as opposed to `train_functions.py`, which both trains and validates during the training loop). It reuses `train_functions.validation_epoch_function` to do the actual forward pass/loss computation, so "test" here is really "one validation-style pass with a different log label" — useful e.g. for evaluating a federated-learning global model on a server-held test set (the docstring explicitly calls out `label='server'` as a FL use case).

## Public API

### `test(training_config: dict, model, test_dataset, label: str = 'test') -> (test_loss: float, log_dict: dict)`
- Builds a `DataLoader` over `test_dataset` using `training_config['batch_size']` (shuffle=True — unusual for a test set, but that's what's implemented), moves `model` to `training_config['device']`, and always uses `torch.nn.CrossEntropyLoss()` as the loss (no configurability).
- Runs `train_functions.validation_epoch_function(...)`, which writes the loss into `log_dict['validation_loss']` by convention; this function then deletes that key and re-inserts it as `log_dict[f'{label}_loss']` so the caller can distinguish e.g. `test_loss` vs `server_loss` vs any other label.
- If `training_config['measure_metrics_during_training']` is `True`, also computes classification metrics via `metrics.compute_metrics(...)` and merges them into `log_dict` with the given `label` prefix via `support_training.update_log_dict_metrics`.
- Returns `(test_loss, log_dict)` — note `test_loss` is the raw float, while `log_dict[f'{label}_loss']` holds the same value under the label-specific key.

## Notes
- The docstring literally says `"""TODO Complete docstring"""` on the `test` function itself — treat documented behavior here as provisional/unverified beyond what's described above.
- Does **not** call `support_training.check_training_config` — it assumes `training_config` has already been validated/defaulted (e.g. by a prior call to `train_functions.train` or a direct `check_training_config` call), since it directly indexes `training_config['batch_size']`, `training_config['device']`, `training_config['measure_metrics_during_training']` without guards.
- Unlike `train_functions.train`, there is no wandb integration inside `test()` itself — callers are expected to log `log_dict` themselves (e.g. via `wandb.log(log_dict)`) if desired.
- No model checkpoint loading happens here — `model` is expected to already have the weights you want to evaluate loaded into it before calling `test()`.
