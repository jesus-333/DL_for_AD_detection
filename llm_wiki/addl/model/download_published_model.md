# download_published_model.py

**Path:** `src/addl/model/download_published_model.py`

## Purpose
Thin wrappers around `torch.hub.load` to fetch third-party pretrained architectures (Inception v3, ResNet50, VGG {11,13,16,19} with/without batch-norm, a 3D ResNet, and a brain-MRI segmentation UNet) plus the matching `torchvision.transforms` preprocessing pipeline for each. Used by `vgg_nets.get_vgg` (for VGG) and standalone for one-off experiments with the other architectures (no other file in `addl.model` currently calls the Inception/ResNet/UNet downloaders).

## Public API

### `download_Inception_v3(pretrained: bool = True)`
Loads `pytorch/vision:v0.10.0` `inception_v3` via `torch.hub`. Returns `(model, preprocess_functions)` where `preprocess_functions` comes from `get_preprocess_functions_config_for_specific_model('inception')` fed into `support_model.get_preprocess_functions`.

### `download_resnet50(pretrained: bool = True)`
Loads `NVIDIA/DeepLearningExamples:torchhub` `nvidia_resnet50`. Returns the model only (no preprocessing config defined for it in this file).

### `download_vgg_nets(version: int, batch_norm: bool, pretrained: bool = True)`
Loads `pytorch/vision:v0.10.0` `vgg{11,13,16,19}` (optionally `_bn` suffixed). Raises `ValueError` if `version` isn't one of 11/13/16/19. Returns `(model, preprocess_functions)`; prints a reminder that inputs must be rescaled to `[0, 1]` before normalization. This is the function `addl.model.vgg_nets.get_vgg` calls to obtain the backbone before wrapping it in the project's `VGG` class.

### `download_resnet_3D(pretrained: bool = True)`
Loads `facebookresearch/pytorchvideo` `slow_r50`. Returns the model only; not otherwise referenced elsewhere in `addl`.

### `download_UNet(pretrained: bool = True)`
Loads `mateuszbuda/brain-segmentation-pytorch` `unet` (fixed `in_channels=3, out_channels=1, init_features=32`) — a brain-MRI abnormality segmentation model, not a classifier. Not otherwise referenced elsewhere in `addl`; present for potential future segmentation experiments.

### `get_preprocess_functions_config_for_specific_model(model_name: str, add_ToTensor_transformation: bool = False) -> dict`
Hardcoded ImageNet-style normalization configs (`mean`, `std`, `resize_size`, `center_crop_size`) for `'inception'` (299/299) and `'vgg'` (256/224), consumed by `support_model.get_preprocess_functions`. Raises `ValueError` for any other `model_name` (note: the error message's listed valid values contains a typo, `'incpetion'`, inherited from the `implemented_model` list literal — cosmetic only). `add_ToTensor_transformation=True` adds `add_ToTensor_transformation: True` to the returned dict.

## Notes
- All downloads require network access to the respective hub repos (`torch.hub.load`) — will fail offline/air-gapped unless the hub cache is already populated.
- Only the VGG path is actually wired into the rest of `addl.model` today; Inception/ResNet50/ResNet3D/UNet downloaders are standalone utilities not consumed by `vgg_nets.py`, `demnet.py`, `vit.py`, `swin.py`, or `vav.py`.
- `pretrained` parameter is accepted but `download_resnet50`'s docstring/usage doesn't surface an equivalent preprocessing config — callers must build their own normalization if needed.
