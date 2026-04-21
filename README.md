# ZedProfiler

[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)](#quality-gates)

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
import os
import pathlib
import pandas as pd

from zedprofiler.IO.loading_classes import ImageSetConfig, ImageSetLoader, ObjectLoader
from zedprofiler.featurization.areasizeshape import compute as compute_areasizeshape

image_set_path = pathlib.Path(
    os.path.expanduser(
        "~/mnt/bandicoot/NF1_organoid_data/data/NF0014_T1/zstack_images/C2-1/"
        )
    ).resolve(strict=True)
mask_set_path = pathlib.Path(
    os.path.expanduser(
        "~/mnt/bandicoot/NF1_organoid_data/data/NF0014_T1/segmentation_masks/C2-1/"
        )
    ).resolve(strict=True)
image_set_config = ImageSetConfig(
    image_set_name="test_set",
    raw_image_key_name=["AGP"],
    mask_key_name=["Nuclei"],
)
image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    mask_set_path=mask_set_path,
    anisotropy_spacing=(1,0.1,0.1),
    channel_mapping={
        "DNA": 405,
        "ER": 488,
        "AGP": 555,
        "Mito": 640,
        "Organoid": "organoid_",
        "Cell": "cell_",
        "Nuclei": "nuclei_",
        "Cytoplasm": "cytoplasm_",
    },
    config=image_set_config
)

object_loader = ObjectLoader(
    image_set_loader=image_set_loader,
    channel_name=image_set_config.raw_image_key_name[0],
    compartment_name=image_set_config.mask_key_name[0],
)
area_dict_df = pd.DataFrame(compute_areasizeshape(
    image_set_loader=image_set_loader,
    object_loader=object_loader,
))
area_dict_df
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
