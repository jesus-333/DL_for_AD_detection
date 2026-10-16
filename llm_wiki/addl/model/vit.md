# vit.py

**Path:** `src/addl/model/vit.py`

## Purpose
Thin, config-driven wrapper around the **`timm`** library's model zoo to create Vision Transformer (ViT) models. `timm` is a hard dependency for this file — import fails loudly (`ImportError` with an install hint) if it isn't installed, unlike the optional-dependency pattern used in `demnet.py`. This module is also reused as the per-direction slice encoder inside `vav.py` (`vit_encoder` wraps `timm_vit`).

## Public API

### `class timm_vit(torch.nn.Module)`
Constructor: `config: dict`. Only `model_name` and `num_classes` are strictly required (`self.required_parameters`); every other config key is passed straight through as `**kwargs` to `timm.create_model(model_name=..., num_classes=..., **kwargs)` — i.e. **this class does not hardcode or validate ViT-specific hyperparameters** (depth, heads, embed dim, etc.); whatever is in the config dict (besides `model_name`/`num_classes`) is forwarded verbatim to `timm`, so correctness of those extra keys depends entirely on what the chosen `timm` model accepts.
- `check_config` performs one piece of naming normalization: if the config uses this project's own convention `input_size`/`input_channels`, they are renamed (`.pop`'d and reinserted) to timm's expected `img_size`/`in_chans` — this keeps the project's config vocabulary consistent across its own model files (`demnet` also uses `input_size`/`input_channels`) while still being compatible with `timm.create_model`'s actual kwarg names.
- `forward(x)`: delegates directly to `self.model(x)`.
- `classify(x, return_prob=False)`: standard softmax/argmax wrapper (same pattern as other models).

### `get_vit(config: dict) -> (timm_vit, preprocess_functions | None)`
Builds a `timm_vit(config)`. If `config['preprocess']` is truthy, resolves timm's own data config via `resolve_data_config({}, model=model.model)` and builds a transform with `create_transform(**data_config)` — i.e. preprocessing here comes from **timm's own registry for the chosen architecture**, not from `support_model.get_preprocess_functions`/`download_published_model`'s hardcoded mean/std configs used elsewhere in this package. If `config['preprocess']` is falsy, returns `(model, None)`. **Note: `config['preprocess']` is read unconditionally with no presence check — a config missing this key raises `KeyError`.**

### `get_defaul_config(vit_name: str)`
Stub — body is `if vit_name == 'vit_tiny_patch16_224': pass`, docstring is literally `"""TODO"""`. **Not implemented**; calling it does nothing and returns `None` regardless of input.

## Notes
- **`get_defaul_config` is a no-op stub** (note also the function name typo `defaul` instead of `default`) — any caller expecting default ViT configs from this function will get nothing; defaults currently live only in the TOML config files (e.g. `scripts_sh_V2/config/model_vit.toml`, `model_vit_micro.toml`, `model_vit_tiny.toml`).
- Because unknown config keys pass straight to `timm.create_model`, a typo'd config key will surface as a `timm`/`TypeError` error far from this file rather than a clear validation message — unlike `demnet.py`/`vgg_nets.py`'s explicit `check_*_config` functions, `vit.py` has no equivalent full-schema validator.
- `config['preprocess']` presence is assumed, not checked — pass it explicitly (`True`/`False`) in any config dict used with `get_vit`.
- Relevant config files: `scripts_sh_V2/config/model_vit.toml`, `model_vit_micro.toml`, `model_vit_tiny.toml`.
