# client.py

**Path:** `src/addl/federated/client.py`

## Purpose

Defines the Flower federated-learning **client** used by this project: a thin adapter between Flower's `NumPyClient` API and this repo's own centralized training code in `addl.training` (`train_functions.train`). Each simulated/real FL client wraps one model instance plus its own local train/validation datasets, and on each `fit()` call runs one (local) multi-epoch training loop using the exact same training routine used for centralized training, then reports weights + metrics back to the server/strategy (in `server.py` / `server_backup.py`).

This is the client-side half of the FL protocol; the companion server/strategy code lives in `server.py`, and the weight (de)serialization helpers it relies on live in `support_federated_generic.py`.

## Public API

### `class flower_client_v1(flwr.client.NumPyClient)`
Flower client implementation.

- `__init__(self, model, train_dataset, validation_dataset, training_config)`
  - Stores the model and the two dataset splits, and a `training_config` dict (same shape as used by `addl.training.train_functions.train`).
  - Reads `training_config['client_id']` (expected to be injected by the caller's `client_fn`, not normally present in the TOML config file itself — see e.g. `client_fn_demnet` in `scripts_python/training_FL/*/client_app.py`). If absent, defaults to `-1` and prints a warning.
  - **Side effect:** mutates `training_config['path_to_save_model']` in place, appending `/client_{client_id}` — so each client writes checkpoints to its own subfolder, and this field must not be reused by other clients/the server after construction (each client gets its own mutated copy of the dict).
  - Creates that save-path folder if it doesn't exist (`os.makedirs`).
  - Defaults `training_config['print_var']` to `False` if not set.
- `get_weights(self, config) -> List[np.ndarray]`: delegates to `support_federated_generic.get_weights(self.model)`.
- `fit(self, parameters, config) -> (weights, num_samples, metrics_dict)`: the core FL round for this client.
  - Loads the server-provided `parameters` into the model via `support_federated_generic.set_weights`.
  - Merges the Flower-provided `config` dict (built server-side by an `on_fit_config_fn`, if any — otherwise empty) into `self.training_config`: `config['lr']` is special-cased into `training_config['optimizer_config']['lr']`; any other key in `config` that already exists in `training_config` overwrites it (keys not already present in `training_config` are silently ignored).
  - Runs local training via `train_functions.train(self.training_config, self.model, self.train_dataset, self.validation_dataset)` — this is the *same* function used for centralized (non-FL) training, so client-side training config/behavior stays in sync with the centralized path by construction.
  - Converts the returned per-epoch `training_metrics` into a flat scalar dict via `convert_training_metrics_for_upload` (Flower requires scalar metric values), and adds `client_id` and `epochs` to it.
  - **Optional early-stop reload:** if `training_config['use_weights_with_lower_validation_error']` is true, it reloads `model_BEST.pth` from `path_to_save_model` (the lowest-validation-error checkpoint saved during `train_functions.train`) before returning weights — so the weights sent back to the server for aggregation may be the early-stopped weights rather than the final-epoch weights.
  - Returns `(get_weights(model), len(train_dataset), converted_training_metrics)` — the standard Flower `fit()` return contract.
- `convert_training_metrics_for_upload(self, training_metrics) -> dict`: Flower's `fit()`/aggregation machinery requires the metrics dict to contain scalars only, but `train_functions.train` returns per-epoch lists (e.g. loss/accuracy per epoch). This flattens each list-valued metric `m` into one scalar key per epoch — `f'{m}:{i}'` for `i` in `range(epochs)` — plus a duplicate `f'{m}:END'` holding the last epoch's value (for easy final-metric lookup). The `confusion_matrix` metric is explicitly skipped (not flattened/uploaded). The inverse of this transformation (re-assembling the per-epoch arrays) is implemented server-side in `support_federated_server.extract_metric_from_log_dict`.

## Notes

- `__dir__(self)` is very likely a **bug**: it's almost certainly meant to be `__del__` (destructor), since its body moves dataset tensors to CPU and deletes `self.train_dataset`/`self.validation_dataset` as a cleanup/memory-management step. As written, `__dir__` is the dunder method Python calls for introspection (e.g. `dir(obj)`), so this cleanup code does not run on client teardown/garbage collection the way it appears to intend — it would only run if something explicitly calls `dir()` on the client instance, and even then it returns `None` instead of a list of attribute names (which would itself break well-behaved `dir()` usage). Worth flagging/fixing in the code itself, not just the docs, if client memory usage is ever an issue.
- `evaluate()` is commented out / explicitly not implemented: evaluation is handled entirely inside `train_functions.train`'s own validation loop, not as a separate Flower `evaluate` round-trip for this client.
- `training_config['client_id']` is expected to be set by the caller (e.g. the `client_fn` in a Flower `ClientApp`, as in `scripts_python/training_FL/*/client_app.py`) — it is **not** a standard key of the training TOML configs under `addl.training`. If a future caller forgets to set it, clients silently default to id `-1` (collision risk if multiple clients do this) and print a warning rather than failing.
- Depends on `addl.dataset` objects supporting `.data.get_device()` / `.move_data_and_labels_to_device(...)` (used, bugged, in `__dir__`) — see `addl/dataset/dataset.py` for the actual dataset classes.
