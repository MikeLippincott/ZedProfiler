# Create and sync the local uv environment with all project dependency groups

set shell := ["bash", "-cu"]

env:
    uv sync --group dev --group docs --group notebooks

# Re-sync environment after dependency changes

sync:
    uv sync --group dev --group docs --group notebooks

# Run tests with coverage gate from pyproject settings

test:
    uv run python -m pytest

# Generate coverage reports (terminal, XML, and HTML)

coverage:
    uv run python -m pytest --cov=zedprofiler --cov-report=term-missing --cov-report=xml --cov-report=html --cov-fail-under=85

# Update README coverage badge from the generated coverage.xml report

coverage-badge:
    uv run python scripts/update_coverage_badge.py --coverage-file coverage.xml --readme README.md

# Run coverage check and refresh coverage badge in one command

coverage-check: coverage
    just coverage-badge

# Run lint checks

lint:
    uv run ruff check .

lint-fix:
    uv run ruff check . --fix

# Build Sphinx docs with docs dependencies

docs:
    cd docs && uv run --group docs sphinx-build src build

# Explicit docs build target (alias) that others can call

docs-build:
    cd docs && uv run --group docs sphinx-build src build

# Serve built docs as static files on port 8000

# This will build first, then serve the generated HTML directory

docs-serve: docs-build
    python -m http.server 8000 --directory docs/build

# Live-reload docs while editing (requires `sphinx-autobuild` in the environment)

# Falls back to a helpful error if `sphinx-autobuild` is not available

docs-autobuild:
    cd docs && uv run --group docs python scripts/run_autobuild.py

# Run the full project workflow (env sync, lint, tests, coverage, and docs build)

all: sync lint-fix lint test coverage-check docs
