# __init__.py

**Path:** `src/addl/model/__init__.py`

## Purpose
Empty file. It only marks `addl.model` as a regular Python package so its submodules (`demnet`, `vgg_nets`, `vit`, `vav`, `swin`, `support_model`, `download_published_model`) can be imported as `addl.model.<module>`. It re-exports nothing.

## Public API
None.

## Notes
No package-level re-exports — callers must import each submodule directly (e.g. `from addl.model import demnet`, `from addl.model.vit import get_vit`).
