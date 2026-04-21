"""Tests for featurization modules."""

from __future__ import annotations

import re

import pytest

from zedprofiler.exceptions import ZedProfilerError
from zedprofiler.featurization import (
    colocalization,
    granularity,
    intensity,
    neighbors,
    texture,
)


class TestColocationizationModule:
    """Tests for colocalization featurization module."""

    def test_compute_raises_not_implemented(self) -> None:
        """Test that colocalization.compute raises ZedProfilerError."""
        with pytest.raises(
            ZedProfilerError,
            match=re.escape("colocalization.compute is not implemented yet"),
        ):
            colocalization.compute()


class TestGranularityModule:
    """Tests for granularity featurization module."""

    def test_compute_raises_not_implemented(self) -> None:
        """Test that granularity.compute raises ZedProfilerError."""
        with pytest.raises(
            ZedProfilerError,
            match=re.escape("granularity.compute is not implemented yet"),
        ):
            granularity.compute()


class TestIntensityModule:
    """Tests for intensity featurization module."""

    def test_compute_raises_not_implemented(self) -> None:
        """Test that intensity.compute raises ZedProfilerError."""
        with pytest.raises(
            ZedProfilerError,
            match=re.escape("intensity.compute is not implemented yet"),
        ):
            intensity.compute()


class TestNeighborsModule:
    """Tests for neighbors featurization module."""

    def test_compute_raises_not_implemented(self) -> None:
        """Test that neighbors.compute raises ZedProfilerError."""
        with pytest.raises(
            ZedProfilerError,
            match=re.escape("neighbors.compute is not implemented yet"),
        ):
            neighbors.compute()


class TestTextureModule:
    """Tests for texture featurization module."""

    def test_compute_raises_not_implemented(self) -> None:
        """Test that texture.compute raises ZedProfilerError."""
        with pytest.raises(
            ZedProfilerError,
            match=re.escape("texture.compute is not implemented yet"),
        ):
            texture.compute()
