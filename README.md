# ZedProfiler

[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](#quality-gates)

CPU-first 3D image feature extraction toolkit for high-content and high-throughput image-based profiling.

This repository is used for image-based feature extraction of objects in 3D microscopy images.
In this use case we extract freatures from single cells in 3D volumetric microscopy images.
We developmed ZedProfiler to be used on high-content and high-throughput microscopy images, which are often large in size and require efficient processing.
ZedProfiler is extensible to any fluorescence microscopy image modality, and is designed to be modular.

## Install environment (development)

```bash
just env
```

## Quick usage (API shape)

```python
from zedprofiler import colocalization  # promoted convenience import
from zedprofiler.featurization import texture  # stays in feature namespace

# Implementation is scaffolded in PR 1 and will be completed in module PRs.
result = colocalization.compute()
texture_result = texture.compute()
```

## Data Contract

Where:

- `x` is the width of the image in pixels
- `y` is the height of the image in pixels
- `z` is the depth of the image in pixels
- `c` is the number of channels

Though I have reservations about width, height, and depth being used to describe the dimensions of an image.
Different fields use different dimensions for different meanings.
We use `x` and `y` to refer to the same dimensions captured in a 2D image, and `z` to refer to the "depth" dimension in a 3D image if looking down into the image stack.
The `x`, `y`, and `z` dimensions are less description and more absolute while `depth` is relative to angle of observation.

Accepted image formats (order matters):

- Single channel: `(z, y, x)`
- Multi channel: `(c, z, y, x)`

## Quality Gates

We lint and format code with our pre-commit configuration.

## Roadmap

The roadmap is maintained in `ROADMAP.md`.
