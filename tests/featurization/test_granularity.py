import re
from types import SimpleNamespace

import numpy as np
import pytest
from _pytest.capture import CaptureFixture

from zedprofiler.featurization.granularity import (
    _fix_scipy_ndimage_result,
    _subsample_3d,
    _upsample_3d,
    compute_granularity,
)

EXPECTED_THREE_SCALES = 3
EXPECTED_TWO_SCALES = 2


def _make_loader(image: np.ndarray, labels: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(image=image, label_image=labels)


def test_fix_scipy_ndimage_result_handles_scalar_and_sequence() -> None:
    scalar_result = _fix_scipy_ndimage_result(3.14)
    sequence_result = _fix_scipy_ndimage_result([1.0, 2.0])

    np.testing.assert_array_equal(scalar_result, np.array([3.14]))
    np.testing.assert_array_equal(sequence_result, np.array([1.0, 2.0]))


def test_subsample_3d_returns_copy_when_factor_is_one() -> None:
    data = np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4)
    out = _subsample_3d(data, np.array(data.shape, dtype=float), subsample_factor=1.0)

    np.testing.assert_array_equal(out, data)
    assert out is not data


def test_subsample_3d_reduces_shape_for_fractional_factor() -> None:
    data = np.arange(4 * 4 * 4, dtype=float).reshape(4, 4, 4)
    new_shape = np.array(data.shape, dtype=float) * 0.5

    out = _subsample_3d(data, new_shape, subsample_factor=0.5)

    assert out.shape == (2, 2, 2)


def test_upsample_3d_restores_requested_shape() -> None:
    data = np.arange(2 * 2 * 2, dtype=float).reshape(2, 2, 2)

    out = _upsample_3d(
        data=data,
        subsampled_shape=np.array([2.0, 2.0, 2.0]),
        original_shape=(4, 4, 4),
    )

    assert out.shape == (4, 4, 4)


@pytest.mark.parametrize(
    ("kwargs", "error_text"),
    [
        ({"subsample_size": 0.0}, "subsample_size must be in (0, 1]"),
        ({"image_sample_size": 0.0}, "image_sample_size must be in (0, 1]"),
        ({"radius": 0}, "radius must be positive"),
        ({"granular_spectrum_length": 0}, "granular_spectrum_length must be positive"),
    ],
)
def test_compute_granularity_validates_inputs(
    kwargs: dict[str, float | int],
    error_text: str,
) -> None:
    image = np.ones((4, 4, 4), dtype=float)
    labels = np.ones((4, 4, 4), dtype=int)
    loader = _make_loader(image, labels)

    with pytest.raises(ValueError, match=re.escape(error_text)):
        compute_granularity(loader, **kwargs)


def test_compute_granularity_returns_empty_measurements_when_no_objects() -> None:
    image = np.ones((6, 6, 6), dtype=float)
    labels = np.zeros((6, 6, 6), dtype=int)
    loader = _make_loader(image, labels)

    result = compute_granularity(
        loader,
        radius=1,
        granular_spectrum_length=2,
        subsample_size=1.0,
        image_sample_size=1.0,
        verbose=False,
    )

    assert result == {"object_id": [], "feature": [], "value": []}


def test_compute_granularity_generates_measurements_for_objects() -> None:
    image = np.zeros((7, 7, 7), dtype=float)
    image[2:5, 2:5, 2:5] = 20.0
    labels = np.zeros((7, 7, 7), dtype=int)
    labels[2:5, 2:5, 2:5] = 1
    loader = _make_loader(image, labels)

    result = compute_granularity(
        loader,
        radius=1,
        granular_spectrum_length=3,
        subsample_size=1.0,
        image_sample_size=1.0,
        verbose=False,
    )

    assert result["object_id"] == [1, 1, 1]
    assert result["feature"] == [1, 2, 3]
    assert len(result["value"]) == EXPECTED_THREE_SCALES


def test_compute_granularity_with_subsampling_mask_and_verbose(
    capsys: CaptureFixture[str],
) -> None:
    image = np.zeros((8, 8, 8), dtype=float)
    image[2:6, 2:6, 2:6] = 30.0
    labels = np.zeros((8, 8, 8), dtype=int)
    labels[2:6, 2:6, 2:6] = 1
    mask = np.ones((8, 8, 8), dtype=bool)
    mask[0, :, :] = False
    loader = _make_loader(image, labels)

    result = compute_granularity(
        loader,
        radius=1,
        granular_spectrum_length=2,
        subsample_size=0.75,
        image_sample_size=0.5,
        mask_threshold=0.5,
        verbose=True,
        image_mask=mask,
    )

    captured = capsys.readouterr()

    assert "Subsampled image" in captured.out
    assert "Background removed via tophat filter." in captured.out
    assert result["object_id"] == [1, 1]
    assert result["feature"] == [1, 2]
    assert len(result["value"]) == EXPECTED_TWO_SCALES
