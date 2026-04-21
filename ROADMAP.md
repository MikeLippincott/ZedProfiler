# ZedProfiler Roadmap

This file is for planning of new modules, features, and PRs that are needed the release(s) of ZedProfiler.
The roadmap is organized into phases, with each phase containing a set of PRs that build upon each other.
The roadmap is intended to be a living document and may be updated as needed.

## Scope for current release

### In scope

- [ ] Handcrafted featurization modules:
  - [ ] AreaSizeShape
  - [ ] Colocalization
  - [ ] Intensity
  - [ ] Granularity
  - [ ] Neighbors
  - [ ] Texture
- [ ] Featurization helper utilities needed by the modules
- [ ] Full test suite with coverage gate (>=85%)
- [ ] Sphinx documentation
- [ ] Example notebooks for module usage
- [ ] Our [RFC2119-specification](./docs/src/Feature_Naming_Convention.md) driven feature naming policy

## Architecture and Product Decisions

- [ ] Linux-first, CPU-only, low RAM usage for the current release.
- [ ] File-format agnostic APIs operating on NumPy arrays.
- [ ] Input contracts:
  - [ ] Single-channel arrays: (z, y, x)
  - [ ] Multi-channel arrays: (c, z, y, x)
- [ ] Modules are intentionally small and composable for external parallel orchestration.

## PR Plan

### Phase 1: Foundation (PR 1-3)

1. PR 1: Packaging and environment baseline

- [ ] Python package scaffold, uv dependency management, version metadata 0.0.1, lint/test tooling, CI skeleton.
- [ ] Linux support and CPU-only scope statements in metadata and docs.

2. PR 2: Core data model and API contracts

- [ ] Canonical input contracts, loader interfaces, common error types.
- [ ] Return schema contract (required keys, types, deterministic ordering).

3. PR 3: RFC2119 naming specification and validators

- [ ] Port and adapt naming conventions into this repository.
- [ ] Add runtime and CI naming validation helpers and conformance tests.

### Phase 2: Feature modules and tests (PR 4-9)

4. PR 4: AreaSizeShape module and tests

   - [x] Implement module

1. PR 5: Colocalization module and tests

   - [x] Implement module

1. PR 6: Intensity module and tests

   - [x] Implement module

1. PR 7: Granularity module and tests

   - [x] Implement module

1. PR 8: Neighbors module and tests

   - [x] Implement module

1. PR 9: Texture module and tests

   - [x] Implement module

### Phase 3: Integration, docs, release (PR 10-13)

10. PR 10: Integration matrix and parallelization guidance

- [ ] Cross-module integration tests and explicit non-goal docs for internal parallelization.

11. PR 11: Example notebooks and public dataset references

- [ ] One notebook per module plus one end-to-end chaining notebook.

12. PR 12: Sphinx docs, logo, API docs

- [ ] API pages, architecture and scope pages, docs build checks in CI.

13. PR 13: Release v0.0.1 and publish workflow

- [ ] Changelog, semantic tag, PyPI publish automation, README install updates.

## Verification Gates

- [ ] Run full unit and integration tests on Linux with coverage >=85%.
- [ ] Run naming validation tests for all emitted feature names.
- [ ] Build Sphinx docs in CI with warnings treated as errors.
- [ ] Execute example notebooks in a clean environment.
- [ ] Validate install/import from both wheel and sdist.
- [ ] Perform release dry-run before publishing.

## v0.0.2 Backlog (Planned)

- [ ] Nucleocentric featurization.
- [ ] Optional mask output mode.
- [ ] Expanded benchmark strategy.

## v0.0.3 Backlog (Planned)

- [ ] Potential Arrow Flight handoff experiments.
