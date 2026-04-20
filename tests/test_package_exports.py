r"""Tests for package export ergonomics."""

import zedprofiler
from zedprofiler import colocalization
from zedprofiler.featurization import texture


def test_top_level_promoted_module_import() -> None:
    """Promoted modules are available from package top-level."""
    assert colocalization is zedprofiler.colocalization
    assert colocalization.__name__ == "zedprofiler.featurization.colocalization"


def test_lower_namespace_import_still_available() -> None:
    """Lower-level namespace imports continue to work."""
    assert texture.__name__ == "zedprofiler.featurization.texture"
