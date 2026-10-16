# dataset.py

**Path:** `src/addl/dataset/dataset.py`

## Purpose
Defines the core in-memory `torch.utils.data.Dataset` implementations used to feed MRI volumes/slices to PyTorch training loops once the raw data has already been converted into tensors (as opposed to `dataset_png.py`, which reads from PNG files on disk, or `dataset_HDF5.py`, which reads lazily from an HDF5 file). This module is the one used when the whole tensor (or a memory-mapped `.pt` file) can be handled directly, e.g. after conversion via `support_dataset_ADNI.get_dataset_V2` or the `npy-to-pt`/`pt-to-npy` CLI tools in `cli_scripts`. It underlies both centralized and federated training code paths since whichever process builds the `DataLoader` just wraps data/labels tensors with one of these classes.

## Public API

### `MRI_dataset(torch.utils.data.Dataset)`
- Wraps a full in-memory tensor (or numpy array) of shape `N x D x H x W` plus a matching `labels` array/list.
- `__init__(data, labels, preprocess_functions=None, print_var=True)`: validates `data.shape[0] == len(labels)`, converts both to `torch.Tensor`. If `preprocess_functions` is given, it is applied **once, to the whole dataset, at construction time** (not per-sample like torchvision datasets normally do) — this is a deliberate perf choice to avoid repeating preprocessing every epoch, but it means the stored `self.data` is already preprocessed and any raw access bypasses re-applying transforms.
- `__getitem__`/`__len__`: standard indexing into `self.data`/`self.labels`.
- `move_data_and_labels_to_device(device)`: moves both tensors to a device in place (used instead of per-batch `.to(device)` to keep everything resident on GPU for small datasets).
- `visualize_sample(idx)`: matplotlib preview for a single 2D sample (channel-last imshow). Raises if `data.shape[1] >= 3`, so it only works for true 2D (non-volumetric) samples.
- `add_channel_dimension_for_MRI_volumes()`: converts `N x D x H x W` → `N x 1 x D x H x W` in place, to match PyTorch's `(N, C, D, H, W)` 3D-conv convention. No-op guard/check — calling it twice will double-insert a dimension.
- `check_single_sample_V1(idx, figsize=None)` / `check_single_sample_V2(idx, figsize=None)`: grid-plot vs. interactive-slider visualization of all depth slices of one 3D sample, used to sanity-check that slice ("depth map") ordering is correct after conversion. V1/V2 require `data.shape[1] > 3` (i.e. real 3D volumes).

### `MRI_dataset_mmap(MRI_dataset)`
- Same conceptual dataset but backed by a memory-mapped `torch.load(..., mmap=True)` tensor file instead of an in-memory array, for datasets too large to fit in RAM.
- `__init__(path_to_tensor_file, labels, preprocess_functions=None, print_var=True)`: loads the tensor lazily via mmap; does **not** apply `preprocess_functions` at construction (unlike the parent class) — see `__getitem__`.
- `__getitem__(idx)`: applies `preprocess_functions` per-sample, at access time, since the data is memory-mapped and not materialized in memory. **Caution (latent bug risk)**: the ternary `(self.preprocess_functions(self.data[idx]) if ... else self.data[idx]), self.labels[idx]` — operator precedence means the tuple is `(conditional_expr, labels[idx])`, which is correct, but it's easy to misread; worth double-checking if modifying this line.
- `move_data_and_labels_to_device(device)`: overridden — re-`torch.load`s the mmap file with `map_location=device` rather than calling `.to(device)` on the mmap tensor (which would materialize the whole thing); only labels get a plain `.to(device)`.
- Inherits `visualize_sample`, `check_single_sample_V1/V2`, `add_channel_dimension_for_MRI_volumes` etc. from `MRI_dataset`, but NOT `__init__`/`__getitem__`/`move_data_and_labels_to_device` which are overridden.

## Notes
- Both classes assume `D` (depth/slice) is the second axis after batch, i.e. PyTorch 3D-conv convention `(N, C, D, H, W)` once `add_channel_dimension_for_MRI_volumes` has been called.
- `TODO` left in `add_channel_dimension_for_MRI_volumes` for an optional shape check before unsqueezing.
- No handling of dtype casting beyond `torch.asarray` — callers are responsible for ensuring `data` is already float/normalized before construction (see `support_dataset_ADNI.get_dataset_V2`'s `apply_rescale` handling).
- Tight coupling with `support_dataset_ADNI.get_dataset_V2`, which is the main producer of `MRI_dataset` instances for ADNI tensor-file-based training.
