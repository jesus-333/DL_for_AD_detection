# __init__.py

**Path:** `src/addl/cli_scripts/__init__.py`

## Purpose
Empty package marker. It makes `addl.cli_scripts` an importable subpackage; it contributes no symbols, re-exports, or logic of its own. Each sibling module (`nii_to_hdf5`, `pt_to_npy`, `stats`, `wandb_scripts`) is imported explicitly where needed (e.g. from `addl/cli.py` as `from .cli_scripts import <module>`), so there is nothing to look up at the package level.

## Public API
None.

## Notes
No notes — file is intentionally empty.
