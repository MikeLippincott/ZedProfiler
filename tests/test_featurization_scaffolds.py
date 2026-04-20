"""Tests for featurization scaffold placeholder behavior."""

from __future__ import annotations

from types import ModuleType

import pytest

from zedprofiler.exceptions import ZedProfilerError
from zedprofiler.featurization import (
    areasizeshape,
    colocalization,
    granularity,
    intensity,
    neighbors,
    texture,
)


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (areasizeshape, "areasizeshape.compute is not implemented yet"),
        (colocalization, "colocalization.compute is not implemented yet"),
        (granularity, "granularity.compute is not implemented yet"),
        (intensity, "intensity.compute is not implemented yet"),
        (neighbors, "neighbors.compute is not implemented yet"),
        (texture, "texture.compute is not implemented yet"),
    ],
)
def test_scaffold_compute_raises_not_implemented(
    module: ModuleType,
    message: str,
) -> None:
    """Each scaffolded compute function should raise a clear placeholder error."""
    with pytest.raises(ZedProfilerError, match=message):
        module.compute()
