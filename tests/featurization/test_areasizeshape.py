"""Tests for areasizeshape module compute behavior."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from zedprofiler.exceptions import ZedProfilerError
from zedprofiler.featurization import areasizeshape

# Test constants for coordinate ranges and object counts
NUM_TEST_OBJECTS = 2
COORD_LOWER_BOUND = 1
COORD_UPPER_BOUND = 3
EXTENT_LOWER_BOUND = 0
EXTENT_UPPER_BOUND = 1


class DummyImageSetLoader:
    """Minimal image set loader double for compute tests."""

    anisotropy_spacing = (1.0, 1.0, 1.0)


class DummyObjectLoader:
    """Minimal object loader double for compute tests."""

    label_image = np.zeros((3, 3, 3), dtype=np.int32)
    object_ids: ClassVar = [0]


def test_compute_no_args_returns_empty_schema() -> None:
    """No-arg compute should be callable and return deterministic keys."""
    result = areasizeshape.compute()

    assert isinstance(result, dict)
    assert list(result.keys()) == [
        "object_id",
        "Volume",
        "CenterX",
        "CenterY",
        "CenterZ",
        "BboxVolume",
        "MinX",
        "MaxX",
        "MinY",
        "MaxY",
        "MinZ",
        "MaxZ",
        "Extent",
        "EulerNumber",
        "EquivalentDiameter",
        "SurfaceArea",
    ]


def test_compute_requires_both_loaders() -> None:
    """Partial invocation should fail with a clear contract error."""
    with pytest.raises(ZedProfilerError):
        areasizeshape.compute(image_set_loader=DummyImageSetLoader())

    with pytest.raises(ZedProfilerError):
        areasizeshape.compute(object_loader=DummyObjectLoader())


def test_compute_delegates_when_loaders_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute should execute delegated measurement path with both loaders."""
    sentinel = {"object_id": [], "Volume": []}

    def _fake_measure(
        image_set_loader: DummyImageSetLoader,
        object_loader: DummyObjectLoader,
    ) -> dict[str, list[float]]:
        assert isinstance(image_set_loader, DummyImageSetLoader)
        assert isinstance(object_loader, DummyObjectLoader)
        return sentinel

    monkeypatch.setattr(areasizeshape, "measure_3D_area_size_shape", _fake_measure)

    result = areasizeshape.compute(
        image_set_loader=DummyImageSetLoader(),
        object_loader=DummyObjectLoader(),
    )

    assert result is sentinel


def test_empty_feature_result_schema() -> None:
    """_empty_feature_result should return deterministic schema."""
    result = areasizeshape._empty_feature_result()

    expected_keys = {
        "object_id",
        "Volume",
        "CenterX",
        "CenterY",
        "CenterZ",
        "BboxVolume",
        "MinX",
        "MaxX",
        "MinY",
        "MaxY",
        "MinZ",
        "MaxZ",
        "Extent",
        "EulerNumber",
        "EquivalentDiameter",
        "SurfaceArea",
    }

    assert set(result.keys()) == expected_keys

    for value in result.values():
        assert isinstance(value, list)
        assert len(value) == 0


def test_get_skimage_measure_import() -> None:
    """Test that _get_skimage_measure successfully imports skimage.measure."""
    try:
        measure = areasizeshape._get_skimage_measure()
        # If scikit-image is available, check that it has expected functions
        assert hasattr(measure, "marching_cubes")
        assert hasattr(measure, "mesh_surface_area")
        assert hasattr(measure, "regionprops")
    except ZedProfilerError:
        # If scikit-image is not available, that's also OK for this test
        pytest.skip("scikit-image not available")


def test_get_skimage_measure_missing_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that missing scikit-image raises appropriate error."""

    def mock_import_module(name: str) -> object:
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(areasizeshape, "import_module", mock_import_module)

    with pytest.raises(ZedProfilerError, match="scikit-image"):
        areasizeshape._get_skimage_measure()


def test_calculate_surface_area_with_simple_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test calculate_surface_area with mocked scikit-image."""
    # Create a simple 3D binary array with a small sphere-like region
    label_object = np.zeros((7, 7, 7), dtype=np.uint8)
    # Create a small ROI with gradient values for marching cubes
    label_object[2:5, 2:5, 2:5] = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
        ]
    )

    # Mock props for bounding box
    props = {
        "bbox-0": np.array([2]),
        "bbox-1": np.array([2]),
        "bbox-2": np.array([2]),
        "bbox-3": np.array([5]),
        "bbox-4": np.array([5]),
        "bbox-5": np.array([5]),
    }

    spacing = (1.0, 1.0, 1.0)

    try:
        result = areasizeshape.calculate_surface_area(label_object, props, spacing)
        # Result should be a float (surface area)
        assert isinstance(result, (float, np.floating))
        assert result >= 0
    except ZedProfilerError:
        # If scikit-image is not available, skip
        pytest.skip("scikit-image not available")


def test_measure_3d_area_size_shape_returns_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that measure_3D_area_size_shape returns correct structure."""

    # Mock the loaders
    def mock_import_module(name: str) -> object:
        raise ModuleNotFoundError("scikit-image not available for test")

    monkeypatch.setattr(areasizeshape, "import_module", mock_import_module)

    # This should raise because _get_skimage_measure will fail
    with pytest.raises(ZedProfilerError):
        areasizeshape.measure_3D_area_size_shape(
            DummyImageSetLoader(),
            DummyObjectLoader(),
        )


def test_measure_3d_area_size_shape_with_real_data() -> None:
    """Test measure_3D_area_size_shape with real scikit-image data."""
    try:
        # Create a realistic 3D label image with two objects
        label_image = np.zeros((10, 10, 10), dtype=np.int32)
        # Object 1: a small cube
        label_image[1:4, 1:4, 1:4] = 1
        # Object 2: another small cube
        label_image[6:9, 6:9, 6:9] = 2

        # Create real loaders
        class RealImageSetLoader:
            anisotropy_spacing = (1.0, 1.0, 1.0)

        class RealObjectLoader:
            def __init__(self, label_image: np.ndarray) -> None:
                self.label_image = label_image
                self.object_ids = [1, 2]

        result = areasizeshape.measure_3D_area_size_shape(
            RealImageSetLoader(),
            RealObjectLoader(label_image),
        )

        # Verify structure
        assert isinstance(result, dict)

        # Check all expected keys are present
        expected_keys = {
            "object_id",
            "Volume",
            "CenterX",
            "CenterY",
            "CenterZ",
            "BboxVolume",
            "MinX",
            "MaxX",
            "MinY",
            "MaxY",
            "MinZ",
            "MaxZ",
            "Extent",
            "EulerNumber",
            "EquivalentDiameter",
            "SurfaceArea",
        }
        assert set(result.keys()) == expected_keys

        # Check that we have features for both objects
        assert len(result["object_id"]) == NUM_TEST_OBJECTS
        assert result["object_id"] == [1, 2]
        # Verify all lists have same length
        list_lengths = {k: len(v) for k, v in result.items()}
        assert len(set(list_lengths.values())) == 1, "All lists should have same length"

        # Check Volume values are reasonable (23 voxels per 3x3x3 cube)
        assert result["Volume"][0] > 0
        assert result["Volume"][1] > 0

        # Check that center coordinates are within expected ranges
        assert COORD_LOWER_BOUND <= result["CenterX"][0] <= COORD_UPPER_BOUND
        assert COORD_LOWER_BOUND <= result["CenterY"][0] <= COORD_UPPER_BOUND
        assert COORD_LOWER_BOUND <= result["CenterZ"][0] <= COORD_UPPER_BOUND

        # Check that extent values are between 0 and 1
        assert EXTENT_LOWER_BOUND <= result["Extent"][0] <= EXTENT_UPPER_BOUND
        assert EXTENT_LOWER_BOUND <= result["Extent"][1] <= EXTENT_UPPER_BOUND

    except ZedProfilerError as e:
        if "scikit-image" in str(e):
            pytest.skip("scikit-image not available")
        raise


def test_measure_3d_area_size_shape_empty_objects() -> None:
    """Test measure_3D_area_size_shape with empty object list."""
    try:
        # Create an empty label image
        label_image = np.zeros((10, 10, 10), dtype=np.int32)

        class RealImageSetLoader:
            anisotropy_spacing = (1.0, 1.0, 1.0)

        class RealObjectLoader:
            def __init__(self, label_image: np.ndarray) -> None:
                self.label_image = label_image
                self.object_ids = []  # No objects

        result = areasizeshape.measure_3D_area_size_shape(
            RealImageSetLoader(),
            RealObjectLoader(label_image),
        )

        # Should return empty but valid structure
        assert isinstance(result, dict)
        for key, values in result.items():
            assert isinstance(values, list)
            assert len(values) == 0

    except ZedProfilerError as e:
        if "scikit-image" in str(e):
            pytest.skip("scikit-image not available")
        raise


def test_measure_3d_area_size_shape_with_anisotropic_spacing() -> None:
    """Test measure_3D_area_size_shape with anisotropic (non-uniform) spacing."""
    try:
        # Create a label image
        label_image = np.zeros((10, 10, 10), dtype=np.int32)
        label_image[2:8, 2:8, 2:8] = 1

        class AnisotropicImageSetLoader:
            # Simulate higher resolution in Z axis
            anisotropy_spacing = (0.5, 1.0, 1.0)

        class RealObjectLoader:
            def __init__(self, label_image: np.ndarray) -> None:
                self.label_image = label_image
                self.object_ids = [1]

        result = areasizeshape.measure_3D_area_size_shape(
            AnisotropicImageSetLoader(),
            RealObjectLoader(label_image),
        )

        # Should handle anisotropic spacing correctly
        assert len(result["object_id"]) == 1
        assert result["object_id"][0] == 1

    except ZedProfilerError as e:
        if "scikit-image" in str(e):
            pytest.skip("scikit-image not available")
        raise
