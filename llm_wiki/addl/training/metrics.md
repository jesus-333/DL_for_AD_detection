# metrics.py

**Path:** `src/addl/training/metrics.py`

## Purpose
Computes classification metrics (accuracy, Cohen's kappa, sensitivity, specificity, F1) for a trained model against a `torch.utils.data.DataLoader`. This is the evaluation building block used both by `train_functions.py` (during the train/validation loop, when `training_config['measure_metrics_during_training']` is `True`) and by `test_functions.py` (for a final test-set evaluation). It is independent of whether training was centralized or federated — the federated server/client code (`addl.federated`) also relies on the same metric computation when it builds config dicts that funnel into `train_functions`/`test_functions`.

All metrics assume a **multiclass classification** setup (the AD/MCI/CN-style diagnosis categories), with integer class labels.

## Public API

### `compute_metrics(model, loader, device) -> dict`
High-level entry point: runs the model over every batch in `loader` to get predicted vs. true labels (via `compute_label`), then calls `compute_metrics_from_labels`. Returns the metrics dict described below. This is what `train_functions.train()` and `test_functions.test()` call each epoch.

### `compute_label(model, loader, device) -> (true_label: torch.Tensor, predict_label: torch.Tensor)`
Puts the model in `eval()` mode, moves it to `device`, iterates the loader, and calls `model.classify(x)` (NOT `model(x)`) to get hard predicted labels — so every model class in `addl.model` is expected to expose a `.classify(x)` method distinct from its forward pass (`.classify` is assumed to return class indices, not logits). Labels are concatenated and moved to CPU.

### `compute_metrics_from_labels(true_label, predict_label) -> dict`
Returns a dict with keys `accuracy`, `cohen_kappa`, `sensitivity` (recall, `average='weighted'`), `specificity` (via `compute_specificity_multiclass`), `f1` (`average='weighted'`). Confusion matrix computation is commented out ("For now the computation/upload is suspended") — not part of the returned dict despite appearing in some other functions' expectations (see Notes).

### `compute_specificity_multiclass(true_label, predict_label, weight_sum=True) -> float`
One-vs-rest specificity averaged across classes found in `true_label` (`set(true_label)`), weighted by per-class support when `weight_sum=True` (the default), otherwise unweighted (`weight_list` of 1s).

### `compute_specificity_binary(true_label, predict_label) -> float`
Binary specificity = TN / (TN + FP), computed from `sklearn.metrics.confusion_matrix`. Guards against division by zero (returns `0.0` if `TN + FP == 0`). Assumes the confusion matrix is 2x2 with class `0` as "negative" in position `[0, 0]`/`[0, 1]` — only valid when called with binarized (one-vs-rest) labels as done by `compute_specificity_multiclass`.

### `compute_multiclass_confusion_matrix(true_label, predict_label) -> np.ndarray`
Manually builds a **hardcoded 4x4** confusion matrix (assumes exactly 4 classes, labels 0–3) and row/column-normalizes by total sample count (not per-row). Currently unused by `compute_metrics_from_labels` (commented out there) — only reachable if called directly.

### `compute_metrics_given_path(model, loader_list, path, device='cpu') -> dict`
Batch-evaluates a directory of saved checkpoints: lists every file in `path`, loads it as a state dict (`torch.load(..., map_location='cpu')`) into `model`, then for every loader in `loader_list` unpacks `compute_metrics(...)` as a 6-tuple `(accuracy, cohen_kappa, sensitivity, specificity, f1, confusion_matrix)` and appends to a `metrics_per_file` dict (including a `confusion_matrix` key).

## Notes
- **Bug risk / inconsistency:** `compute_metrics_given_path` unpacks the return of `compute_metrics` as a 6-tuple, but `compute_metrics` actually returns a **dict** with only 5 keys (no `confusion_matrix`, since that computation is commented out in `compute_metrics_from_labels`). Calling `compute_metrics_given_path` as written will raise an error (can't unpack a dict of 5 keys into 6 variables, and dict iteration order isn't tuple-unpacking semantics anyway). This function appears unused/untested — treat it as broken until fixed.
- `compute_label` requires every model passed in to implement `.classify(x)` (see `addl.model.support_model` / individual model files) — a model lacking this method will break every metric computation.
- No multi-label support: all metrics assume single-label multiclass classification with integer-encoded classes.
- Depends only on `numpy`, `torch`, `sklearn.metrics`, `os` — no coupling to config/toml structures (unlike `support_training.py`).
