# ZedProfiler

CPU-first handcrafted 3D featurization toolkit for Linux HPC and cloud execution.

## Install environment (development)

```bash
just env
```

## Quick usage (API shape)

```python
from zedprofiler.featurization import colocalization

# Implementation is scaffolded in PR 1 and will be completed in module PRs.
result = colocalization.compute()
```

## Data Contract (initial)

Accepted image formats:

- Single channel: `(z, y, x)`
- Multi channel: `(c, z, y, x)`

## Quality Gates

- Lint: `ruff`
- Tests: `pytest`
- Coverage: enforced at `>=85%`
- Docs: `sphinx-build -W`

## Roadmap

The canonical roadmap is maintained in `ROADMAP.md`.
