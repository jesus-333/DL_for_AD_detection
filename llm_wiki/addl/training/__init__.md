# __init__.py

**Path:** `src/addl/training/__init__.py`

## Purpose
Empty file. It exists only to mark `addl.training` as a Python package/subpackage so that its sibling modules (`metrics`, `support_training`, `test_functions`, `train_functions`) can be imported as `addl.training.<module>`. It contributes no code, re-exports, or package-level API of its own.

## Public API
None.

## Notes
No `__all__`, no re-exports. Importing `addl.training` directly gives you nothing useful — callers always import the specific submodule they need (e.g. `from addl.training import train_functions`).
