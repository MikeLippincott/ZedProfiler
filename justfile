set shell := ["bash", "-cu"]

# Create and sync the local uv environment with all project extras.
env:
    uv sync --extra dev --extra docs --extra featurization

# Re-sync environment after dependency changes.
sync:
    uv sync --extra dev --extra docs --extra featurization

# Run tests with coverage gate from pyproject settings.
test:
    uv run python -m pytest

# Run lint checks.
lint:
    uv run ruff check .
