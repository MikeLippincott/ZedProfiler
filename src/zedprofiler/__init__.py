"""Top-level package for ZedProfiler.

Selected featurizers are promoted for convenient imports, while the full
module tree remains available under ``zedprofiler.featurization``.
"""

from zedprofiler._version import __version__

# Re-export commonly used featurization modules at package top-level.
# This keeps imports ergonomic while preserving the canonical nested namespace:
# `zedprofiler.featurization.<module>`.
from zedprofiler.featurization import (
    areasizeshape,
    colocalization,
    granularity,
    intensity,
    neighbors,
    texture,
)

__all__ = [
    "__version__",
    "areasizeshape",
    "colocalization",
    "granularity",
    "intensity",
    "neighbors",
    "texture",
]
