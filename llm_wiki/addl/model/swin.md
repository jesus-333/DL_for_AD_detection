# swin.py

**Path:** `src/addl/model/swin.py`

## Purpose
Wraps `torchvision`'s Swin Transformer (v1 and v2, sizes tiny/small/base) into a classifier with the project's common model interface (`classify`, `set_training_mode`, etc.), mirroring what `vgg_nets.py` does for VGG. Per project history this file was renamed from an earlier `transformer.py` — no other file in the current tree references the old name, so the rename appears complete. Uses `torchvision.models.swin_*` pretrained backbones (ImageNet weights via `*_Weights.DEFAULT`); a `timm` import is present but commented out, unused.

## Public API

### `class swin_transformer_classifier(torch.nn.Module)`
Constructor args: `num_classes: int` (required, must be `> 0`), `version: int = 2` (1 or 2), `size: str = 'tiny'` (`'tiny'|'small'|'base'`).
- Builds the torchvision Swin encoder via `set_swin_model()` → `get_swin_model_v1`/`get_swin_model_v2`, then replaces `encoder.head` with a fresh `torch.nn.Linear(encoder.head.in_features, num_classes)` — i.e. it reuses torchvision's own classification head slot rather than adding a separate `self.classifier` (a `self.classifier` attribute is assigned but then immediately unused/shadowed by `self.encoder.head = classifier`; `forward` never references `self.classifier`, only `self.encoder`). Also stores `self.transforms` (the ImageNet weight's own preprocessing pipeline, from `weight_obj.transforms()`) and `self.apply_transforms_to_input` (default `False` — the transforms are **not** applied automatically unless this flag is set).
- `forward(x)`: input `B x C x H x W` → if `apply_transforms_to_input`, applies `self.transforms` first, then passes through `self.encoder` (head included) → output `B x num_classes`.
- `classify(x, return_prob=False)`: standard softmax-probabilities-or-argmax wrapper, same pattern as other models in this package.
- `set_training_mode(training_mode: int)`: 0 = train all; 1 = freeze everything except `encoder.head` (the classifier). Only two modes (unlike VGG/DemNet's 3–4 modes). Raises `ValueError` for anything else.
- `set_version_and_size(version, size, update_model=True)` / `set_version(version, update_model=True)` / `set_size(size, update_model=True)`: mutate `self.version`/`self.size` after validating via `check_version_and_size`, and rebuild the encoder (`set_swin_model()`) when `update_model=True`. **Caution**: calling these after construction rebuilds the backbone from scratch (fresh pretrained weights) and does *not* re-attach the custom classification head sized for `num_classes` — only `set_swin_model()` is called, which re-derives `self.encoder` from `get_swin_model_v{1,2}` with torchvision's original (ImageNet, 1000-class) head. Changing version/size post-construction therefore silently reverts the output layer to 1000 classes; `__init__`'s head-replacement logic is not re-run.
- `check_version_and_size(version, size)`: raises `ValueError` if `version not in [1, 2]` or `size not in ['tiny', 'small', 'base']`.

### `get_swin_model_v1(size: str = 'tiny', return_transforms: bool = True)` / `get_swin_model_v2(...)`
Thin wrappers over `torchvision.models.swin_{t,s,b}` / `swin_v2_{t,s,b}`, always using `*_Weights.DEFAULT` (i.e. always pretrained — there's no "from scratch" option; a `TODO add possibility to use model not pretrained` is noted in the class docstring). Returns `(model, transforms)` if `return_transforms=True`, else just `model`.

## Notes
- **Head-reset bug on `set_version`/`set_size`/`set_version_and_size`**: as above — switching version/size after construction loses the custom `num_classes` head. Any future training-mode "model surgery" code should rebuild/re-wrap the classifier head after calling these setters, or this should be fixed upstream.
- `self.classifier` is dead/unused state (assigned once, never read).
- No config-dict constructor exists for this model unlike `demnet`/`vgg_nets`/`vit` (those take a single `config: dict`); `swin_transformer_classifier` takes plain keyword args instead — if this project wants config-driven construction consistency across `addl.model`, this is the odd one out (check `scripts_sh_V2/config/model_swin.toml` for how callers currently bridge TOML config → these constructor args).
