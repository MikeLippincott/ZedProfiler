"""Smoke tests for top-level package access."""

import zedprofiler
from zedprofiler import colocalization


def test_package_version_is_exposed() -> None:
    """The package should expose a string version."""
    assert isinstance(zedprofiler.__version__, str)
    assert zedprofiler.__version__


def test_promoted_feature_module_is_accessible() -> None:
    """Promoted feature modules should be importable from top-level."""
    assert colocalization.__name__ == "zedprofiler.featurization.colocalization"
