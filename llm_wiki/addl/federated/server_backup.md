# server_backup.py

**Path:** `src/addl/federated/server_backup.py`

## Purpose

An **older/alternate implementation** of the same FL server strategy as `server.py`: a class also named `fed_avg_with_wandb_tracking`, subclassing `flwr.server.strategy.FedAvg`, adding wandb tracking to federated averaging. Functionally it covers the same responsibilities (wandb run/artifact setup, per-round metric/weight logging, periodic checkpointing, centralized evaluation logging), but implements most of that logic **inline** in this file rather than delegating to `support_federated_server.py`.

Given the identical class name and overlapping responsibility with `server.py`, this file reads as a **retained previous version** of the server strategy — likely kept for reference while the "real" implementation in `server.py` was refactored to extract shared logic into `support_federated_server.py`. See `server.md` for the presumed-current version.

## Public API

### `class fed_avg_with_wandb_tracking(flwr.server.strategy.FedAvg)`
Same constructor signature and config contract as the version in `server.py` (`model`, `all_config` with `server_config`/`dataset_config`/`model_config`/`training_config`), but with these differences:
- `__init__`: explicitly checks `wandb_installed` and raises `ImportError` if `wandb` is missing (more defensive than `server.py`'s unconditional `wandb.watch` call). Inlines the wandb run/artifact creation logic that `server.py` delegates to `support_federated_server.setup_wand`. Contains an explicit `# TODO test this features` comment noting that `wandb.watch`'s gradient/weight logging does not appear to work correctly, with a guess at why (possibly `wandb.log()` with an explicit `step`, or the fact that no backprop happens on this process so no "update" is detected).
- `aggregate_fit`: inlines (rather than calling into `support_federated_server`) the parameter conversion (`flwr.common.parameters_to_ndarrays`), periodic model saving via `support_federated_generic.save_model_weights`, per-client metric extraction/plotting (`support_federated_server.extract_metric_from_log_dict` + `create_and_log_wandb_metric_plot_separately`, still used from the support module here), and end-of-training wandb run closure via its own method `end_wandb_run_and_log_artifact` (duplicated from `support_federated_server.end_wandb_run_and_log_artifact`).
- `evaluate`: same conditional-on-`centralized_evaluation` structure as `server.py`, but inlines the wandb logging and run-closing logic instead of calling `support_federated_server.evaluate_log`.
- `end_wandb_run_and_log_artifact(self)`: local copy of the logic also present as a standalone function in `support_federated_server.py`.

## Notes

- **Do not use alongside `server.py`** — both modules define a class of the same name (`fed_avg_with_wandb_tracking`) implementing the same role; importing/using both in the same run is confusing at best and could mask which strategy is actually active. Treat this file as historical/backup, not a second supported strategy.
- If `server.py`'s strategy ever needs debugging, this file is a useful reference for "what the un-refactored, more defensive (ImportError-checked) version looked like," including the TODO about `wandb.watch` gradient logging not working — that TODO was not carried over into `server.py`'s docstring/comments, so it's easy to lose track of if this file is deleted without transferring the note.
- Depends on: `support_federated_generic.set_weights`, `support_federated_generic.save_model_weights`, `support_federated_server.extract_metric_from_log_dict`, `support_federated_server.create_and_log_wandb_metric_plot_separately`.
