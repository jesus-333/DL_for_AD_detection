# support_federated_server.py

**Path:** `src/addl/federated/support_federated_server.py`

## Purpose

Server/strategy-specific helper functions backing `server.py`'s `fed_avg_with_wandb_tracking` strategy: wandb run/artifact lifecycle management, per-round aggregation + logging of both server-side and per-client metrics, periodic/final model checkpointing, and plot generation (matplotlib and native wandb plots) for per-client training curves. This is the module `server.py` delegates almost all of its non-trivial logic to; `server_backup.py` contains an older version with much of this logic inlined instead of factored out here (though it still reuses a couple of functions from this file, e.g. `extract_metric_from_log_dict`).

## Public API

### `extract_metric_from_log_dict(log_dict: dict) -> (list, list)`
- Inverse of `client.flower_client_v1.convert_training_metrics_for_upload`: Flower's metrics dict must be flat/scalar, so the client flattens per-epoch metric arrays into keys like `f'{metric}:{epoch}'`. This function reconstructs, for each metric name found in `log_dict` (detected by presence of a `'{name}:0'` key), the full per-epoch numpy array, returning `(metrics_values_list, metrics_name_list)` (parallel lists, same order).

### `compute_lr_after_round(lr_value: float, server_round: int)`
- **Stub / not implemented** — body is just `pass`. Not called anywhere else in this file; presumably a placeholder for a future server-driven LR-schedule-across-rounds feature.

### `setup_wand(wandb_config: dict, config_to_backup: dict) -> (wandb_run, model_artifact)`
- Initializes a `wandb.init(...)` run (project/entity/name/notes pulled defensively from `wandb_config`, since optional TOML keys may be entirely absent rather than `None` — TOML has no null type) and, if `wandb_config['log_model_artifact']` is true, a `wandb.Artifact` for model weights (else `model_artifact=None`). Raises `ImportError` up front if `wandb` isn't installed. This is what `server.py.__init__` calls (unlike `server_backup.py`, which inlines the same logic).

### `aggregate_fit_log(aggregated_parameters, aggregated_metrics, server_round, results, model, wandb_run, count_rounds, rounds_to_save_model, num_rounds, all_config, model_artifact=None, metrics_to_log_from_clients=None) -> (aggregated_parameters, aggregated_metrics, model_weights, count_rounds)`
- The main per-round orchestration function called from `server.py.aggregate_fit`. Delegates to `aggregate_fit_log_clients_data` (if parameters present) and `aggregate_fit_log_server_metrics` (if metrics present, else prints a warning), increments `count_rounds`, and — if this was the final round (`count_rounds == num_rounds`) and centralized evaluation is **not** enabled (so there's no later `evaluate()` call to close the run) — saves a final `model_round_END.pth` checkpoint and closes the wandb run via `end_wandb_run_and_log_artifact`.
- Note the `all_config` parameter: the docstring explicitly explains only 2 of its nested values are actually used, kept as a single param purely because this logic used to live inline in the strategy class (which already held `all_config` as `self.all_config`) — a readability/refactor trade-off worth knowing if extending this function.

### `aggregate_fit_log_clients_data(aggregated_parameters, model, count_rounds, rounds_to_save_model, all_config, results, wandb_run, model_artifact=None, metrics_to_log_from_clients=None) -> model_weights`
- Converts `aggregated_parameters` (Flower `Parameters`) to numpy weights, loads them into `model` via `support_federated_generic.set_weights`, and — every `rounds_to_save_model` rounds (if `rounds_to_save_model > 0`) — saves a checkpoint (`model_round_{count_rounds}.pth`) and optionally uploads it as a wandb artifact file.
- If `metrics_to_log_from_clients` is set (`'all'` or a list of metric names from `{accuracy, cohen_kappa, sensitivity, specificity, f1}`), iterates over each client's result, extracts their per-epoch metrics via `extract_metric_from_log_dict`, and logs one wandb line-plot per metric per client via `create_and_log_wandb_metric_plot_separately`.

### `aggregate_fit_log_server_metrics(aggregated_metrics: dict, wandb_run, count_rounds: int) -> None`
- Extracts only the `*:END` keys from `aggregated_metrics` (final-epoch value per metric, per the client's flattening convention) and logs them to wandb at `step=count_rounds`.

### `evaluate_log(test_metrics_dict: dict, wandb_run, count_rounds: int, num_rounds: int, model_artifact=None) -> (float, dict)`
- Called from `server.py.evaluate` when centralized evaluation is enabled: logs `test_metrics_dict` to wandb, and if this is the final round, closes the run via `end_wandb_run_and_log_artifact`. Despite the `-> (float, dict)` type hint, the function body does not actually return anything (implicit `None`) — the type hint appears aspirational/stale relative to the implementation.

### `end_wandb_run_and_log_artifact(wandb_run, model_artifact=None)`
- Logs `model_artifact` to the run (if provided) and calls `wandb_run.finish()`. Duplicated (not shared) in `server_backup.py` as a method of the strategy class there.

### `create_matplotlib_metric_plot(metrics_to_plot_list: list, training_epochs, metrics_name_list: list)`
- Builds a matplotlib figure/axes plotting each metric array against `training_epochs` on one shared axis. Forces the `'agg'` (non-interactive) backend explicitly, with an inline comment explaining this is required because matplotlib is not thread-safe and Flower's simulation runs server callbacks off the main thread. Returns `(fig, ax)`; caller is responsible for closing the figure.

### `create_and_log_matplotlib_metric_plot(metrics_to_plot_list, training_epochs, metrics_name_list, client_id, count_rounds, wandb_run)`
- Builds the plot via `create_matplotlib_metric_plot` (note: calls it with an extra `client_id` positional arg that `create_matplotlib_metric_plot`'s signature does not accept — see Notes, this looks like a latent bug / dead code path) and logs the resulting figure object to wandb under key `client_{id}/{metric_name_or_'all'}_round_{round}_plt`, with `commit=False`. Not currently called from the main `aggregate_fit_log_clients_data` path (which calls `create_and_log_wandb_metric_plot_separately` instead) — appears to be an alternative/legacy plotting path left in place but unused.

### `create_and_log_wandb_metric_plot_separately(metrics_to_plot_list, training_epochs, metrics_name_list, client_id, count_rounds: int, wandb_run)`
- The actually-used plotting path (called from `aggregate_fit_log_clients_data` and from `server.py`/`server_backup.py` directly). For each metric, builds a `wandb.Table` and logs a native wandb line plot under `client_{id}/{metric_name}_round_{round}` (the `/` groups plots into a wandb UI section per client — documented inline). Logs with `commit=False` (accumulates; actual upload happens on the next `commit=True` log call, e.g. in `aggregate_fit_log_server_metrics`).

### `create_and_log_wandb_metric_plot_together(metrics_to_plot_list, training_epochs, metrics_name_list, client_id, count_rounds: int, wandb_run)`
- Alternative to the "separately" variant: builds one wandb Table with all requested metrics as columns and logs a single combined line plot. Marked in `server_backup.py`'s caller comment as **"not producing the plot I want"** — i.e. known not to work as intended; not used in the active `aggregate_fit_log_clients_data` path.

## Notes

- `create_and_log_matplotlib_metric_plot` calls `create_matplotlib_metric_plot(metrics_to_plot_list, training_epochs, metrics_name_list, client_id)` — four positional args — but `create_matplotlib_metric_plot` is defined with only three parameters. Calling the former as written would raise a `TypeError`. Since it isn't on the active code path (the server code calls `create_and_log_wandb_metric_plot_separately` instead), this bug is currently latent/dormant rather than triggered in normal use — flag before resurrecting this plotting path.
- `evaluate_log`'s return type hint (`-> (float, dict)`) does not match its actual (implicit `None`) return — don't rely on a return value from this function.
- Several functions here (`aggregate_fit_log*`) are essentially FedAvg-round orchestration extracted out of the strategy class so both `server.py` and (partially) `server_backup.py` can share them — when modifying aggregation behavior, check both strategy files to see which one a given training script actually instantiates.
