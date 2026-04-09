"""Core data contracts shared across featurizers.

The package accepts either:
- Single-channel 3D arrays shaped (z, y, x)
- Multi-channel 4D arrays shaped (c, z, y, x)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

DimensionOrder = Literal["zyx", "czyx"]


@dataclass(frozen=True)
class ImageArrayContract:
    """Document expected dimensionality and ordering for input arrays."""

    dimensions: int
    order: DimensionOrder


FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]

SINGLE_CHANNEL_CONTRACT = ImageArrayContract(dimensions=3, order="zyx")
MULTI_CHANNEL_CONTRACT = ImageArrayContract(dimensions=4, order="czyx")
