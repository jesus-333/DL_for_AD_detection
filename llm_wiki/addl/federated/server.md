# server.py

**Path:** `src/addl/federated/server.py`

## Purpose

Defines the Flower **strategy** (server-side aggregation logic) used by this project: `fed_avg_with_wandb_tracking`, a subclass of Flower's built-in `FedAvg` strategy that adds Weights & Biases (`wandb`) experiment tracking on top of standard federated averaging. This is the server-side counterpart to `client.py`; it consumes the per-client metrics produced by `client.flower_client_v1.fit` (via `convert_training_metrics_for_upload`) and the shared helpers in `support_federated_generic.py` / `support_federated_server.py`.

This file is the **current/active** server strategy implementation (see Notes on its relationship to `server_backup.py`).

## Public API

### `class fed_avg_with_wandb_tracking(flwr.server.strategy.FedAvg)`
- `__init__(self, model, all_config: dict, *args, **kwargs)`
  - `all_config` must contain `server_config` (with a nested `wandb_config`), `dataset_config`, `model_config`, `training_config` — see the docstring in the code for the full expected key set (`num_rounds`, `num_clients`, `fraction_fit`, `fraction_evaluate`, `keep_labels_proportion`, `centralized_evaluation`, `metrics_to_log_from_clients`, `metrics_plot_backend`, `rounds_to_save_model`, etc.). Several of these keys (e.g. `num_clients`, `fraction_fit`, `fraction_evaluate`, `keep_labels_proportion`) are **not used inside this class** — they only matter to whatever builds the Flower `ServerApp`/simulation (e.g. `scripts_python/training_FL/*/server_app.py`).
  - Delegates wandb run + artifact setup to `support_federated_server.setup_wand(wandb_config, all_config)` (unlike `server_backup.py`, which inlines this logic — see Notes).
  - Stores `count_rounds` (starts at 0), `num_rounds`, `rounds_to_save_model`, `metrics_to_log_from_clients`, and `model`.
  - Calls `wandb.watch(self.model, log="all", log_freq=1, log_graph=True)` (see Notes — flagged elsewhere in the codebase as not fully working).
- `aggregate_fit(self, server_round, results, failures)`: calls `super().aggregate_fit(...)` (standard FedAvg weighted parameter averaging), then delegates **all** post-processing — logging client/server metrics to wandb, periodically saving model weights, handling end-of-training wandb run closure — to `support_federated_server.aggregate_fit_log(...)`. Returns the (possibly re-derived) `aggregated_parameters, aggregated_metrics`.
- `evaluate(self, server_round, parameters)`: only runs if `all_config['server_config']['centralized_evaluation']` is true, in which case it calls `super().evaluate(...)` (which invokes whatever `evaluate_fn` was passed to the strategy at construction) and logs the result via `support_federated_server.evaluate_log(...)`. If centralized evaluation is disabled, it just prints a one-time notice on round 0 and implicitly returns `None` (not a `(loss, metrics)` tuple — see Notes).

## Notes

- **Relationship to `server_backup.py`**: `server_backup.py` defines a class with the *exact same name* (`fed_avg_with_wandb_tracking`) and very similar logic, but implements wandb setup, model-weight saving, and per-client metric plotting **inline** rather than via `support_federated_server` helpers, and contains extra inline commentary/TODOs (e.g. about `wandb.watch` not logging gradients correctly). This strongly suggests `server_backup.py` is an **older version kept as a reference/fallback** while the logic was refactored out into `support_federated_server.py` for this file. Since both define the same class name, **do not import both in the same process** — only one module should be the active strategy implementation. Treat `server.py` as canonical going forward; if behavior diverges from what's expected, cross-check against `server_backup.py` before assuming it's a regression.
- `evaluate()` returns `None` implicitly when `centralized_evaluation` is `False`, whereas Flower's `Strategy.evaluate` contract normally expects either `None` or a `(loss, metrics_dict)` tuple — returning `None` is valid Flower-side (means "skip centralized eval"), but be aware this method has two different return shapes depending on config.
- `wandb.watch(...)` is called unconditionally in `__init__`; if `wandb` is not installed, `wandb_installed` is `False` but this is **not checked** before use in this file (unlike `support_federated_server.setup_wand`, which does raise `ImportError` if `wandb` is missing) — so constructing this strategy without `wandb` installed will raise a `NameError`/`AttributeError` from the unguarded `wandb.watch` call, not the friendlier `ImportError` message defined at import time.
- Depends on: `support_federated_generic.set_weights`, `support_federated_server.setup_wand`, `support_federated_server.aggregate_fit_log`, `support_federated_server.evaluate_log`.
