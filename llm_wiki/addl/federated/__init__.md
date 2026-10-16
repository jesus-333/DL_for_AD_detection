# __init__.py

**Path:** `src/addl/federated/__init__.py`

## Purpose

Package initializer for `addl.federated`. It does not define any federated-learning logic itself; its only job is to fail fast, with a clear message, if the optional federated-learning dependencies are missing. This subpackage wraps the [Flower](https://flower.ai/) framework to implement FL training (client/server/strategy) for the models in `addl.model`, trained on the datasets in `addl.dataset`.

## Public API

This file exposes no classes or functions. It only runs import-time checks.

## Notes

- `import flwr` is required; if missing, it raises `ValueError("To use the federated learning module you need to install flwr. Run 'pip install flwr'.")`. Note this raises on package import, so simply `import addl.federated` (even indirectly, e.g. via `addl.federated.client`) will crash the whole process if `flwr` is not installed — there is no lazy/optional import path for the base package.
- `from flwr.simulation import run_simulation` is treated as a *soft* dependency: if `flwr[simulation]` extras are not installed, it only prints a warning and continues (does not raise). This matches `pyproject.toml`, where `flwr[simulation]` is a top-level dependency, so in practice this soft-fail path is unlikely to trigger unless flwr was installed without the `simulation` extra.
- `run_simulation` is imported here but not re-exported or used elsewhere in this file; it exists purely to validate the import works at package load time.
