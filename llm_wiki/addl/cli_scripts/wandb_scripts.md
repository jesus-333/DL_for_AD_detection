# wandb_scripts.py

**Path:** `src/addl/cli_scripts/wandb_scripts.py`

## Purpose
Backs two CLI commands for interacting with Weights & Biases (wandb) as an external experiment-tracking service: `wandb-get-run-status` and `wandb-download-project-runs-metrics` (both registered in `addl.cli`). This is the only cli_scripts module that talks to a remote service rather than the local filesystem. It depends on `addl.other.wandb_interactions` for the heavier per-run data wrangling (see `llm_wiki/addl/other/wandb_interactions.md`).

## Public API
### `get_run_status(args: argparse.Namespace) -> None`
- Required `args`: `entity`, `project`, `run_id`.
- Calls `wandb.Api().run(f"{entity}/{project}/{run_id}")` and prints `run.state` (e.g. `running`, `finished`, `crashed`) to stdout — this is the entire output contract of the `wandb-get-run-status` CLI command (callers are expected to capture stdout).
- Catches `wandb.errors.CommError` and generic `Exception` separately, printing a message instead of raising — so CLI failures here are **silent** (exit code 0) rather than propagating as CLI errors.

### `download_project_runs_metrics(args: argparse.Namespace) -> None`
- Required `args`: `entity`, `project`; optional `path_save` (defaults to `./wandb_metrics.json`, printed as a notice if omitted).
- Fetches all runs in the project via `wandb.Api().runs(f"{entity}/{project}")`, and for each run:
  - Gets structured run info via `wandb_interactions.get_run_information(run)` (expected to return a dict with at least `'history'` (a DataFrame) and `'name'`).
  - Strips model-parameter columns from the history via `wandb_interactions.remove_parameters_from_history(...)`.
  - Builds a unique key per run as `f"{run_info['name']}_{run.id}"`.
  - Accumulates into `metrics_dict[metric_name][run_key] = list_of_values`, skipping any column starting with `_` (wandb's own bookkeeping columns like `_step`, `_runtime`, `_timestamp`).
- Writes the whole `metrics_dict` as pretty-printed JSON (`indent=4`) to `path_save`.
- Same silent-failure pattern as `get_run_status`: `wandb.errors.CommError` and generic `Exception` are caught and printed, not raised.

## Notes
- Both functions assume a wandb API key / login is already configured in the environment (not validated here — an auth failure would likely surface as a generic exception message).
- The JSON output shape is: `{metric_name: {"<run_name>_<run_id>": [values...]}}` — documented in the function's own docstring with an example.
- No `--debug` flag is read in this file (unlike the other `cli_scripts` modules), so there's no intermediate progress reporting while iterating over potentially many runs.
