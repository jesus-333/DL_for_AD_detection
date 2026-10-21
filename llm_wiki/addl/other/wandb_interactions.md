# wandb_interactions.py

**Path:** `src/addl/other/wandb_interactions.py`

## Purpose
Small helper module for **reading back** data from the Weights & Biases (wandb) API after training runs have happened — as opposed to `addl.training.train_functions`/`addl.training.support_training`, which handle *writing* to wandb during training. This is a post-hoc analysis utility: pull a run's history/summary/config out of wandb into plain Python/pandas structures for downstream analysis (e.g. by scripts under `scripts_python/analysis/` or `scripts_python/plot/`). It sits in `addl.other` rather than `addl.training` presumably because it's about querying/consuming wandb data rather than producing it, and doesn't fit the dataset/model/federated/training taxonomy.

Related but distinct from `addl.cli_scripts.wandb_scripts` (used by the `wandb-get-run-status` / `wandb-download-project-runs-metrics` CLI commands) and from `addl.model.download_published_model` (downloads model artifacts, not run metrics) — this file is lower-level, meant to be imported and called on a `wandb.apis.public.Run` object you already have (e.g. from `wandb.Api().runs(...)`), not a CLI entry point itself.

## Public API

### `get_run_information(run: wandb.apis.public.Run) -> dict`
Extracts and returns `{name, id, history, summary, config}` from a wandb `Run` object:
- `name`/`id`: the run's wandb name/id.
- `history`: `run.history()` as a `pandas.DataFrame` — the full time series of logged metrics (and, if `wandb.watch(..., log="all")` was used during training, also per-parameter histogram data, columns prefixed `parameters/`).
- `summary`: `run.summary._json_dict` — final/summary metric values (uses the private `_json_dict` attribute rather than public API, to avoid wandb's lazy-loading wrapper around large values — this is a deliberate but fragile choice since it reaches into a "private" attribute).
- `config`: `run.config` filtered to drop any key starting with `_` (wandb convention for internal/special config keys).

### `remove_parameters_from_history(history_df: pd.DataFrame) -> pd.DataFrame`
Drops every column whose name starts with `"parameters/"` from a history DataFrame — i.e. strips the per-parameter gradient/weight histograms logged by `wandb.watch(model, log="all")` in `train_functions.train`, leaving just the scalar metrics (losses, accuracy, learning rate, etc.). Useful because those parameter columns can be extremely wide/heavy for a model with many layers.

## Notes
- The bottom of the file (lines ~84–113) is a **commented-out example snippet**, not executable code — a worked example of using `wandb.Api()` to pull `summary`/`config`/`name` for every run in a project (`alberto_zancanaro_academic/demnet_FL_random_partioning_PAPER`) and assemble them into a DataFrame, saved as reference/scratch code rather than deleted. It hints at the project-name convention used for past wandb experiments.
- No error handling around network/auth failures from the wandb API — callers are expected to have `wandb` configured/logged-in already.
- `get_run_information` reaching into `run.summary._json_dict` (a private attribute) means this could break on a wandb client library upgrade that changes internals; there is no public-API equivalent currently used as a fallback.
