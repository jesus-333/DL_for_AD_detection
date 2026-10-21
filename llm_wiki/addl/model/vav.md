# vav.py

**Path:** `src/addl/model/vav.py`

## Purpose
Implements the **Volume-Attention ViT (VAV)** architecture: a model designed to classify whole 3D MRI volumes (not single 2D slices) by running a ViT over the slices along each of the three anatomical directions (axial, sagittal, coronal), then fusing the per-slice embeddings with a Transformer encoder into one volume-level embedding for classification. Depends on `vit.py`'s `timm_vit`. This is one of the project's newer/more complex architectures (added alongside ViT/Swin per project history) aimed specifically at exploiting full-volume 3D context rather than per-slice 2D classification.

## Public API

### `class vav(torch.nn.Module)`
Config dict keys: `use_same_vit_for_all_directions: bool` (share one ViT across axial/sagittal/coronal vs. one per direction), `vit_config: dict` (passed straight to `vit.timm_vit` for each direction's ViT — all three directions always use the *same* `vit_config` even when `use_same_vit_for_all_directions=False`, only the weights differ, not the architecture), `analyze_directions_independently: bool`, `slice_transformer_config: dict` (gets `d_input` injected automatically from `vit_config['embed_dim']`), `num_classes: int`.
- If `analyze_directions_independently=True`: one `slice_transformer_encoder` per direction, outputs concatenated → classifier input dim = `3 * d_model`.
- If `False`: the three ViT-embedding sequences are concatenated along the **sequence dimension** first, then passed through a single shared `slice_transformer_encoder` → classifier input dim = `d_model`.
- `forward(x: torch.Tensor) -> torch.Tensor`: input `B x X x Y x Z` (NIfTI-convention volume — **the code has an explicit unresolved `TODO CHECK IF THE ORDER IS CORRECT`** regarding whether X/Y/Z really map to sagittal/coronal/axial; a wrong mapping wouldn't break correctness, just mislabel which named sub-module (`vit_axial` etc.) processes which anatomical direction). Internally: `vit_axial(x)`, `vit_sagittal(x.permute(0,2,1,3))`, `vit_coronal(x.permute(0,3,1,2))` — each `vit_*` is actually a `vit_encoder` (see below), which itself expects `B x S x H x W`, so the permutes are what turn the 3D volume into a "batch of slices" for each direction. Output: `B x num_classes`.
- `classify(x, return_prob=False)`: standard softmax/argmax wrapper (docstring says input shape `B x C x H x W`, which is stale/copy-pasted from the 2D models — `vav.forward` actually expects the 4D volume shape above).

### `class vit_encoder(torch.nn.Module)`
Wraps `vit.timm_vit` to turn a volume's slices along one direction into a sequence of per-slice embeddings. Forces `config['num_classes'] = 0` so the underlying `timm_vit` has no classification head (timm convention: `num_classes=0` → output is the pooled embedding instead of logits).
- `forward(x)`: input `B x S x H x W` (`S` = number of slices along this direction) → reshapes to `(B*S) x 1 x H x W` to batch all slices through the ViT in one call (documented perf optimization vs. a per-slice loop), then reshapes back to `B x S x D` (`D = vit_config['embed_dim']`).
- `check_reshape_trick(x, use_eval_mode=True, use_double_precision=False)`: a self-test that verifies the batched-reshape forward pass is numerically equivalent to looping slice-by-slice; author notes ~1e-6 difference in float32 (expected due to op-ordering), reduced to 0 with `use_double_precision=True`. Debugging/verification utility, not used in the normal training path.

### `class positional_encoding(torch.nn.Module)`
Standard sinusoidal positional encoding (`sin`/`cos` on `exp(-log(10000)*2i/d_model)` frequencies), registered as a non-learnable buffer (`register_buffer`, moves with `.cuda()` but isn't optimized). `max_len`, `dropout` configurable. `forward(x)` adds the encoding (truncated to `x`'s sequence length) and applies dropout.

### `class slice_transformer_encoder(torch.nn.Module)`
Projects the ViT-embedding sequence (`d_input → d_model` via `input_proj`), adds positional encoding, runs a `torch.nn.TransformerEncoder` (`batch_first=True`, configurable `nhead`, `dim_feedforward`, `dropout`, `activation`, `num_layers`), then mean-pools over the sequence dimension to produce one `B x d_model` embedding per volume/direction. Supports an optional `mask` (`src_key_padding_mask`) — if given, pooling becomes a masked mean (zeroing padded positions, dividing by true sequence lengths) instead of a plain mean; this is for variable slice counts per direction, though no caller in this file currently passes a mask.

### `class slice_transformer_classifier(slice_transformer_encoder)`
A standalone classifier variant (encoder + linear `head`) — **not used by `vav`** (which builds `slice_transformer_encoder` directly and does its own classification head). Appears to be an alternative/earlier usage of the transformer-encoder block as a complete classifier on its own. Note: its `__init__` calls `super().__init__()` with **no arguments**, but `slice_transformer_encoder.__init__` requires a `config` positional argument — **this class is currently broken and will raise `TypeError` if instantiated** (`super().__init__()` needs `config`; it's never passed).

## Notes
- **Bug**: `slice_transformer_classifier.__init__` is broken (`super().__init__()` missing required `config` arg) — do not use without fixing.
- **Unresolved TODO**: axis-to-anatomical-direction mapping in `vav.forward` is explicitly flagged as unverified in the source comment; doesn't affect correctness, only naming/interpretability.
- `vav_config.vit_config['num_classes']` gets force-overwritten to `0` by `vit_encoder` regardless of what's in the TOML — the volume-level `num_classes` lives only in the outer `vav` config, consumed by its own `self.classifier`.
- Relevant config file: `scripts_sh_V2/config/model_vav.toml`.
