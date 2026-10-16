# vgg_nets.py

**Path:** `src/addl/model/vgg_nets.py`

## Purpose
Wraps a torchvision VGG backbone (downloaded via `download_published_model.download_vgg_nets`) into the project's common model interface. One of the earliest/most-used architectures in the repo (VGG training scripts exist in `scripts_sh/train_vgg_*`, `scripts_python/training/vgg_*`, `scripts_python/test_code/train_vgg_*`).

## Public API

### `class VGG(torch.nn.Module)`
Constructor: `model` (a VGG `torch.nn.Module` already downloaded via `download_published_model.download_vgg_nets`), `num_classes: int`, `input_channels: int = 3`.
- Replaces `model.classifier[-1]` with a fresh `Linear(4096, num_classes)` (assumes the standard torchvision VGG classifier shape, where the last `Linear` has `in_features=4096`).
- If `input_channels != 3`, replaces `model.features[0]` (the first conv layer) with a new `Conv2d(input_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))` — enables grayscale (1-channel) input, or >3-channel input (e.g. **stacking multiple MRI slices as separate "color" channels**, per the docstring's own example).
- Stores `features`, `avgpool`, `classifier` from the passed-in `model` as its own submodules (not the whole `model` object itself) and prints a reminder that input must be rescaled to `[0, 1]`.
- `forward(x)`: `features → avgpool → flatten → classifier`, standard torchvision VGG forward.
- `classify(x, return_prob=False)`: standard softmax/argmax wrapper.
- `set_training_mode(training_mode: int)`: always fully freezes first, then selectively unfreezes: 0 = everything; 1 = only `classifier[6]` (final linear); 2 = all of `classifier`; 3 = `features[0]` (input conv) + `classifier[6]` — useful for finetuning a pretrained VGG onto MRI input channels while keeping the pretrained trunk mostly frozen.
- `freeze_model()`: freezes `features`, `avgpool`, `classifier` explicitly.
- `check_freeze_layer()`: prints `name, requires_grad` for every parameter.

### `get_vgg(config: dict, return_preprocess_functions: bool = True)`
Validates `config` via `check_model_config`, then calls `download_published_model.download_vgg_nets(config['version'], config['batch_norm'], pretrained=config['use_pretrained_vgg'])` and wraps the result in `VGG(model, config['num_classes'], config['input_channels'])`.
- Required config keys: `version` (11/13/16/19), `batch_norm` (bool), `num_classes`; `input_channels` optional (default 3, with a printed note if missing); **`use_pretrained_vgg` is read but not validated by `check_model_config`** (missing key will raise `KeyError` at the `download_vgg_nets` call rather than a clean validation error).
- Returns `(model, preprocess_functions)` if `return_preprocess_functions=True`, else just `model`. `preprocess_functions` is the `Resize/CenterCrop/Normalize` `Compose` from `download_published_model`, expecting `[0,1]`-scaled 3-channel-shaped input *before* this transform runs Normalize.

### `check_model_config(config: dict)`
Validates `version` (required, must be in `[11,13,16,19]`), `batch_norm` (required, presence only — not type-checked), `input_channels` (optional, defaulted to 3 in place with a printed note, must be `>0`), `num_classes` (required, must be `>0`). Does **not** validate `use_pretrained_vgg` (see above) — a config-schema gap relative to what `get_vgg` actually consumes.

## Notes
- **Config validation gap**: `use_pretrained_vgg` is required by `get_vgg`'s call into `download_vgg_nets` but is never checked by `check_model_config` — a config missing this key fails late/unclearly (`KeyError`) instead of via the project's usual clean `ValueError` pattern.
- Preprocessing assumes the `download_published_model.get_preprocess_functions_config_for_specific_model('vgg')` config (mean/std ImageNet stats, resize 256/crop 224) — inconsistent with non-3-channel `input_channels` use cases (e.g. feeding stacked MRI slices), where ImageNet normalization stats may not be meaningful; this mismatch is not flagged/handled anywhere in the code.
- Relevant config file: `scripts_sh_V2/config/model_vgg.toml`.
