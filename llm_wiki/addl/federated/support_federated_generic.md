# support_federated_generic.py

**Path:** `src/addl/federated/support_federated_generic.py`

## Purpose

Shared helper functions for the FL module that are **not** specific to the server/strategy (those live in `support_federated_server.py`). Covers three areas: splitting a dataset across simulated clients, converting model weights to/from the `List[np.ndarray]` form Flower's protocol uses, and a generic weighted-average metric-aggregation function usable as a Flower `fit_metrics_aggregation_fn`/`evaluate_metrics_aggregation_fn`. Some functions here are meant to be called from outside `addl` entirely (e.g. `split_data_for_clients` from data-prep scripts), while others (`set_weights`, `get_weights`) are internal plumbing used by `client.py` and `server.py`.

## Public API

### `split_data_for_clients(data, percentage_split_per_client: list, seed: int = None, labels=None, keep_labels_proportion: int = False)`
- Splits `data` (and, if given, `labels`) into `len(percentage_split_per_client)` groups, one per simulated client, according to the given proportions (which must sum to ~1 — checked against a small whitelist of floating-point-safe sums, raising `ValueError` otherwise).
- If `keep_labels_proportion=True`, uses `support_dataset.get_idx_to_split_data_V3` (stratified split preserving class proportions per client) instead of `support_dataset.get_idx_to_split_data_V2` (plain split); requires `labels` to be provided, or raises `ValueError`.
- Returns `(data_per_client, labels_per_client)` if `labels` given, else just `data_per_client`.

### `split_data_for_clients_uniformly(data, num_clients: int, seed: int = None, labels=None, keep_labels_proportion: int = False)`
- Convenience wrapper: builds an equal-proportion split list (`[1/num_clients] * num_clients`) and calls `split_data_for_clients`. Note equal-size splits are not guaranteed to be exact when `keep_labels_proportion=True` and classes are imbalanced / not evenly divisible — documented as a known limitation in the docstring.

### `check_split_correctness(original_data, original_label, data_per_client, labels_per_client)`
- Sanity-check utility: rebuilds a `{data_value: label}` lookup from the original (unsplit) data/labels and verifies every `(data, label)` pair inside `data_per_client`/`labels_per_client` matches. Raises `ValueError` with a descriptive message on mismatch, otherwise just prints `"Everything seems correct"`. Author's own comment acknowledges this is a belt-and-suspenders check rather than one expected to catch real bugs if the split functions are correct.

### `load_data_and_labels(file_path_list_client: str, label_list_int_client: str)`
- Validates that every element of `file_path_list_client` is a string ending in `.jpg`/`.png`/`.jpeg` (so this is **image-path-based data only**, not used for the HDF5/NIfTI pipelines) and that every label is a non-negative integer, then returns the two lists unchanged (effectively just a validator/pass-through). Raises `ValueError` on any violation.

### `set_weights(model, weights: List[np.ndarray])`
- Loads a list of numpy-array weights (as sent over the wire by Flower) into a `torch.nn.Module`'s `state_dict`, with `strict=True`. Adapted from the official Flower PyTorch tutorial, but with a fix: for any state-dict key containing `'num_batches'` (i.e. BatchNorm's `num_batches_tracked` buffers), it keeps the model's **existing** value instead of overwriting with the (incompatible/incorrectly-typed) incoming value — the stock Flower tutorial approach was found not to work for models with BatchNorm layers. Raises (and re-raises, with an extra printed message) `RuntimeError` on architecture mismatch.

### `get_weights(model) -> List[np.ndarray]`
- Inverse of `set_weights`: returns `[v.cpu().numpy() for v in model.state_dict().values()]`. Adapted from the Flower tutorial (signature renamed from `get_parameters(net)`).

### `save_model_weights(model: torch.nn.Module, path_to_save_model: str, filename: str) -> str`
- Saves `model.state_dict()` via `torch.save` to `path_to_save_model/filename`, creating the directory if needed (`os.makedirs(..., exist_ok=True)`). Returns the full save path. Used by both the client (early-stop reload path, indirectly) and server-side strategy/support code for periodic + final checkpointing during FL rounds.

### `weighted_average(metrics_per_client)`
- Intended to be passed as `fit_metrics_aggregation_fn` / `evaluate_metrics_aggregation_fn` to the Flower strategy constructor. Input: a list of `(num_samples_i, metrics_dict_i)` tuples, one per client (Flower's own call convention). Computes, for every metric key present across clients, the sample-count-weighted average (`np.average(..., weights=n_samples_per_client)`). Loosely based on the Flower `advanced-pytorch` example. Returns a single `dict` of aggregated metrics.

## Notes

- This file and `support_federated_server.py` both exist because, per the module docstring here, "the server has its own dedicated support file" — the split here is by **audience** (client-agnostic/generic utilities vs. server/strategy-specific utilities like wandb plotting), not by client-vs-server role exclusively (e.g. `set_weights`/`get_weights`/`save_model_weights` here are used by both client and server code).
- `load_data_and_labels` only validates `.jpg`/`.png`/`.jpeg` paths — if the FL pipeline is ever pointed at the HDF5/NIfTI-based datasets (`addl.dataset.dataset_HDF5`), this validator would reject them; it looks tied specifically to the PNG/Kaggle-style image-path dataset flow (`addl.dataset.dataset_png`, `support_dataset_kaggle`).
- `check_split_correctness` builds a Python `dict` keyed by raw `data` values — this requires `data` elements to be hashable (fine for file-path strings, would break for e.g. raw numpy arrays).
