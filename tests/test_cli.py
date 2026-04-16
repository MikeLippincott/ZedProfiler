"""Package namespace tests replacing obsolete template CLI checks."""

from zedprofiler.featurization import texture


def test_feature_namespace_import() -> None:
    """Lower-level feature namespace remains importable."""
    assert texture.__name__ == "zedprofiler.featurization.texture"
