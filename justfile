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

# Generate coverage reports (terminal, XML, and HTML).
coverage:
    uv run python -m pytest --cov=zedprofiler --cov-report=term-missing --cov-report=xml --cov-report=html --cov-fail-under=85

# Update README coverage badge from the generated coverage.xml report.
coverage-badge:
    uv run python scripts/update_coverage_badge.py --coverage-file coverage.xml --readme README.md

# Run coverage check and refresh coverage badge in one command.
coverage-check: coverage
    just coverage-badge

# Run lint checks.
lint:
    uv run ruff check .

# Build Sphinx docs with docs dependencies.
docs:
    cd docs && uv run --group docs sphinx-build src build

# Run the full project workflow (env sync, lint, tests, coverage, and docs build).
all: sync lint test coverage-check docs
