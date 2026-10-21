# support_model.py

**Path:** `src/addl/model/support_model.py`

## Purpose
Small shared-utility module for the `addl.model` package: activation-function lookup, trainable-parameter counting, and building a `torchvision.transforms.Compose` preprocessing pipeline from a plain config dict. Used by `demnet.py` (activation lookup), `download_published_model.py` (preprocessing config → `Compose`), and implicitly by any model needing standard image preprocessing.

## Public API

### `get_activation(activation_name: dict) -> torch.nn.Module`
Despite the type hint (`dict`, likely a docstring/annotation mistake — it's actually used as a `str`), maps a case-insensitive string to an activation module instance: `'relu'→ReLU`, `'elu'→ELU`, `'selu'→SELU`, `'gelu'→GELU`. Raises `ValueError` for anything else. **Returns a single shared module instance** — if the same `config['activation']` string is reused to build multiple layers (as `demnet.py` does, passing the same `activation` object into `conv_1`, `conv_2`, and the classifier `Sequential`), those layers all share the exact same stateless activation module instance, which is fine since activations here are stateless, but worth knowing if an activation ever gains learnable parameters (e.g. switching to `PReLU` would break this sharing pattern).

### `count_trainable_parameters(layer: torch.nn.Module) -> int`
Sums `p.numel()` over `layer.parameters()` where `p.requires_grad`. Simple helper, not currently called elsewhere in `addl.model` (each model instead defines its own `print_trainable_parameters`-style method).

### `get_preprocess_functions(config: dict = None) -> torchvision.transforms.Compose`
Builds a `Compose` from a config dict, in this fixed order: `Resize(resize_size)` → `CenterCrop(center_crop_size)` → `Normalize(mean, std)` → optionally `ToTensor` **prepended at the front** if `config['add_ToTensor_transformation'] is True`. Each key is optional — only present keys add their transform. Raises `ValueError` if `resize_size`/`center_crop_size` are present but `<= 0`, or if the resulting transform list would be empty. `mean`/`std` must both be present together to add `Normalize` (if only one is given, normalization is silently skipped). Consumed by `download_published_model.py`'s per-model preprocessing configs (inception/vgg) and usable directly by any other model via a hand-built config dict.

## Notes
- No support for extra transform kwargs (e.g. interpolation mode for `Resize`) — noted as a known limitation in the original docstring.
- `get_activation`'s type annotation (`dict`) does not match actual usage (a `str` is expected) — a documentation/annotation bug, not a runtime issue since Python doesn't enforce hints.
