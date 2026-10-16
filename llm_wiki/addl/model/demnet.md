# demnet.py

**Path:** `src/addl/model/demnet.py`

## Purpose
PyTorch implementation of **DEMNET**, a custom CNN architecture for Alzheimer's diagnosis from 2D MR images (paper: https://ieeexplore.ieee.org/abstract/document/9459692). This is one of the main model families trained in this repo (alongside VGG, ViT, Swin, VAV), used in both centralized and federated training scripts.

The model is built entirely from a nested `config: dict` (typically loaded from a `model.toml`, e.g. `scripts_sh_V2/config/model_demnet.toml`/`model_demnet_tiny.toml`). `check_demnet_config`/`check_demnet_block_config` validate the config in place, filling in defaults (with a printed warning) for several optional keys rather than raising. This makes the config contract somewhat implicit — the authoritative list of keys and defaults is these two check functions plus the `demnet`/`demnet_block` docstrings.

Optional dependencies (`pytorch_grad_cam`, `wandb`) are imported in `try/except` at module load; if missing, related methods print a warning and no-op/fail rather than crashing at import time. This means importing `addl.model.demnet` never fails due to these extras, but calling e.g. `compute_gradcam_for_input` without `pytorch-grad-cam` installed silently returns `None`.

## Public API

### `class demnet(torch.nn.Module)`
Full network: `conv_1` → `conv_2` → `pool` → 4× `demnet_block` → (flatten + MLP classifier), with two dropout layers inserted after block 3 and block 4.
- **Config keys** (validated/defaulted by `check_demnet_config`): `input_size` (required, square image side), `input_channels` (default 3), `output_channels_first_layer` (also the in/out channel count for block 1), `num_classes` (required), `activation` (`relu|elu|selu|gelu`, via `support_model.get_activation`), `kernel_size_conv_1`, `kernel_size_conv_2` (required), `batch_norm` (default `True`), `config_demnet_block_1..4` (each validated by `check_demnet_block_config`), `dropout_rate_1`/`dropout_rate_2` (default 0.5, must be in `(0, 1)`), `use_activation_in_classifier` (default `True`), `use_as_features_extractor` (default `False`).
- Classifier is a fixed MLP: `n_input_neurons → 512 → 128 → 64 → num_classes`, where `n_input_neurons` is computed dynamically via `compute_size_after_conv` (runs a dummy forward pass with `use_as_features_extractor=True` to get the conv-output shape — i.e. the constructor **runs the network once** on a zero tensor just to size the classifier).
- `forward(x)`: if `use_as_features_extractor` is `True`, returns the conv-stack feature map (no flatten/classify) — used as a building block for other architectures or embedding extraction.
- `classify(x, return_prob=False)`: convenience wrapper — softmax probabilities (`B x num_classes`) or `argmax` class ids (`B`).
- `set_training_mode(training_mode: int)`: 0 = train everything; 1 = freeze all except the last classifier `Linear` (`classifier[6]`); 2 = freeze all except the whole classifier MLP. Used for fine-tuning a pretrained DEMNET.
- `freeze_model()`: sets `requires_grad=False` on all parameters.
- `compute_gradcam_for_input` / `visualize_gradcame_for_input`: Grad-CAM explainability (XAI), only functional if `pytorch_grad_cam` is installed — note the code references `self.gradcam`, which is never actually set anywhere in this class (the lines that would build it are commented out), so **`compute_gradcam_for_input` will raise `AttributeError` in its current state** even with the dependency installed.
- `print_input_size_after_each_conv_layer`, `print_trainable_parameters`, `check_freeze_layer`, `check_average_parameters_value_per_layer`, `check_average_grad_value_per_layer`, `print_debug`: debugging/introspection helpers (print shapes, parameter counts, per-layer grad/requires_grad/mean stats). Written ad hoc to debug a near-zero-gradient issue the author hit during training.

### `class demnet_block(torch.nn.Module)`
One repeating block: `conv_1_block` (same in/out channels) → activation → `conv_2_block` (in→out channels) → activation → `norm` (BatchNorm2d or Identity) → `MaxPool2d(2)`. Config keys: `input_channels_block`, `output_channels_block`, `kernel_size_conv_1`, `kernel_size_conv_2`, `batch_norm`, plus `activation` (read from the **parent** config dict passed to `demnet_block.__init__`, not validated by `check_demnet_block_config` itself — see Notes).

### `get_model_pretrained_with_kaggle_dataset(version: int, load_early_stop_weights: bool = True, epoch_to_load: int = -1) -> (torch.nn.Module, dict)`
Downloads a DEMNET checkpoint from a W&B artifact (`jesus_333/demnet_training/demnet_training_AD_kaggle_trained:v{version}`), rebuilds a `demnet` from the artifact's stored `model_config`, and loads weights (`model_BEST.pth` for early-stop, or `model_{epoch}.pth` otherwise) onto CPU. Documents specific known artifact versions: v5 = 4-class, v7 = 3-class (VeryMild+Mild merged), v8 = 2-class (all dementia merged). Requires `wandb` installed and network access; raises if `load_early_stop_weights=False` and `epoch_to_load<=0`.

### `check_demnet_config(config: dict)` / `check_demnet_block_config(config: dict)`
In-place config validators. Raise `Exception` (not a specific exception subclass) for missing/invalid required keys; mutate `config` in place to inject defaults for optional keys, printing a `Warning:` line when doing so.

## Notes
- **Side-effecting validation**: both check functions mutate the passed-in `config` dict (adding default keys) — callers should not assume the config they pass in is unchanged after construction.
- **`gradcam` attribute bug**: `compute_gradcam_for_input`/`visualize_gradcame_for_input` reference `self.gradcam`, which is never initialized (the `pytorch_grad_cam.GradCAM(...)` construction lines are commented out in `__init__`-adjacent code) — currently broken regardless of whether `pytorch-grad-cam` is installed.
- **`demnet_block` has no `activation` entry in `check_demnet_block_config`**: the block reads `config['activation']` directly (via `support_model.get_activation`), but the validator for block configs never checks/defaults this key — it only works today because `demnet.__init__` passes the parent's already-validated `activation` string down... actually it does not pass it down automatically; check the exact config wiring before assuming `config_demnet_block_N` dicts carry their own `activation` key (this looks like it may need an `activation` key per-block in the TOML, undocumented by the validator).
- Exceptions raised are plain `Exception(...)`, not domain-specific exception classes — catching narrowly by type is not possible for these validators.
- `use_as_features_extractor` lets a `demnet` instance serve purely as a convolutional feature extractor for composition into larger models.
