# __init__.py

**Path:** `src/addl/__init__.py`

## Purpose

Package marker for the top-level `addl` package (the installable package built by hatchling from `src/addl`, see `pyproject.toml` — `[tool.hatch.build.targets.wheel.sources]` maps `src/` to the wheel root, so `src/addl` becomes the importable `addl` package).

## Public API

None. The file is currently empty — no re-exports, no `__version__`, no package-level docstring.

## Notes

- Because nothing is re-exported here, every subpackage (`addl.dataset`, `addl.federated`, `addl.model`, `addl.training`, `addl.other`, `addl.cli_scripts`) must be imported via its full dotted path (e.g. `from addl.model import demnet`), not via `addl.demnet` or similar shortcuts.
- If the project later wants a `addl.__version__` (e.g. for the `--version` flag already wired in `cli.py`'s pattern, or for `pyproject.toml`'s `version = "0.0.1"`), this is the place it would be defined.
