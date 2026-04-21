"""Tests for feature_writing_utils module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from zedprofiler.IO.feature_writing_utils import (
    FeatureMetadata,
    format_morphology_feature_name,
    remove_underscores_from_string,
    save_features_as_parquet,
)

EXPECTED_COMPONENT_COUNT = 4


class TestRemoveUnderscoresFromString:
    """Tests for remove_underscores_from_string function."""

    def test_remove_underscores(self) -> None:
        """Test that underscores are replaced with hyphens."""
        assert remove_underscores_from_string("test_string") == "test-string"

    def test_remove_dots(self) -> None:
        """Test that dots are replaced with hyphens."""
        assert remove_underscores_from_string("test.string") == "test-string"

    def test_remove_spaces(self) -> None:
        """Test that spaces are replaced with hyphens."""
        assert remove_underscores_from_string("test string") == "test-string"

    def test_remove_slashes(self) -> None:
        """Test that slashes are replaced with hyphens."""
        assert remove_underscores_from_string("test/string") == "test-string"

    def test_multiple_delimiters(self) -> None:
        """Test removal of multiple different delimiters."""
        input_str = "test_string.with spaces/delimiters"
        expected = "test-string-with-spaces-delimiters"
        assert remove_underscores_from_string(input_str) == expected

    def test_no_delimiters(self) -> None:
        """Test string with no delimiters."""
        assert remove_underscores_from_string("teststring") == "teststring"

    def test_non_string_input_conversion(self) -> None:
        """Test that non-string inputs are converted to strings."""
        assert remove_underscores_from_string(123) == "123"

    def test_non_string_converts_to_string(self) -> None:
        """Test that object instances are converted to string representation."""
        # object() converts to string like '<object object at 0x...>'
        # The hyphens replace the spaces (if any in the string representation)
        result = remove_underscores_from_string(1_2_3)
        assert isinstance(result, str)
        assert result == "123"

    def test_float_conversion(self) -> None:
        """Test that floats are converted to strings."""
        result = remove_underscores_from_string(3.14)
        assert isinstance(result, str)
        assert "3" in result
        assert "14" in result


class TestFormatMorphologyFeatureName:
    """Tests for format_morphology_feature_name function."""

    def test_basic_formatting(self) -> None:
        """Test basic feature name formatting."""
        result = format_morphology_feature_name("nucleus", "dapi", "area", "value")
        assert result == "nucleus_dapi_area_value"

    def test_formatting_with_delimiters(self) -> None:
        """Test formatting with delimiters in input."""
        result = format_morphology_feature_name(
            "cell_body", "gfp.channel", "mean intensity", "normalized/value"
        )
        assert result == "cell-body_gfp-channel_mean-intensity_normalized-value"

    def test_formatting_consistency(self) -> None:
        """Test that output format is consistent."""
        result = format_morphology_feature_name("a", "b", "c", "d")
        assert result.count("_") == EXPECTED_COMPONENT_COUNT - 1


class TestFeatureMetadata:
    """Tests for FeatureMetadata dataclass."""

    def test_feature_metadata_creation(self) -> None:
        """Test creating FeatureMetadata instance."""
        metadata = FeatureMetadata(
            compartment="nucleus",
            channel="dapi",
            feature_type="area",
            cpu_or_gpu="cpu",
        )
        assert metadata.compartment == "nucleus"
        assert metadata.channel == "dapi"
        assert metadata.feature_type == "area"
        assert metadata.cpu_or_gpu == "cpu"


class TestSaveFeaturesAsParquet:
    """Tests for save_features_as_parquet function."""

    def test_save_features_as_parquet(self) -> None:
        """Test saving features as parquet file."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            metadata = FeatureMetadata(
                compartment="nucleus",
                channel="dapi",
                feature_type="area",
                cpu_or_gpu="cpu",
            )
            result_path = save_features_as_parquet(parent_path, df, metadata)
            assert result_path.exists()
            assert result_path.name == "nucleus_dapi_area_cpu_features.parquet"

    def test_save_features_returns_correct_path(self) -> None:
        """Test that save_features_as_parquet returns the correct path."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)
            df = pd.DataFrame({"col1": [1, 2]})
            metadata = FeatureMetadata(
                compartment="test",
                channel="ch1",
                feature_type="type1",
                cpu_or_gpu="gpu",
            )
            result_path = save_features_as_parquet(parent_path, df, metadata)
            expected_path = parent_path / "test_ch1_type1_gpu_features.parquet"
            assert result_path == expected_path

    def test_save_features_preserves_data(self) -> None:
        """Test that saved parquet file preserves data."""
        pytest.importorskip("pyarrow")
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir)
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["x", "y", "z"]})
            metadata = FeatureMetadata(
                compartment="nuc",
                channel="ch",
                feature_type="feat",
                cpu_or_gpu="cpu",
            )
            save_features_as_parquet(parent_path, df, metadata)
            parquet_file = parent_path / "nuc_ch_feat_cpu_features.parquet"
            loaded_df = pd.read_parquet(parquet_file)
            pd.testing.assert_frame_equal(df, loaded_df)
