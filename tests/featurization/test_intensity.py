from types import SimpleNamespace

import numpy as np

from zedprofiler.featurization.intensity import compute_intensity, get_outline

EXPECTED_MEASUREMENT_COUNT = 42


def test_get_outline_marks_boundaries_per_slice() -> None:
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[0, 1:3, 1:3] = True
    mask[1, 0:2, 0:2] = True

    outline = get_outline(mask)

    assert outline.shape == mask.shape
    assert outline.dtype == bool
    assert outline[0].any()
    assert outline[1].any()


def test_compute_intensity_returns_measurements_for_objects() -> None:
    image = np.zeros((3, 3, 3), dtype=float)
    image[0, 0, 0] = 1.0
    image[0, 0, 1] = 2.0
    image[0, 1, 0] = 3.0
    image[0, 1, 1] = 4.0
    image[1, 1, 1] = 5.0
    image[1, 1, 2] = 6.0

    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0, 0:2, 0:2] = 1
    labels[1, 1, 1:3] = 2

    loader = SimpleNamespace(
        image=image,
        label_image=labels,
        object_ids=[1, 2],
        channel="channel_a",
        compartment="nuclei",
    )

    result = compute_intensity(loader)

    expected_features = {
        "IntegratedIntensity",
        "MeanIntensity",
        "StdIntensity",
        "MinIntensity",
        "MaxIntensity",
        "LowerQuartileIntensity",
        "UpperQuartileIntensity",
        "MedianIntensity",
        "MassDisplacement",
        "MeanAbsoluteDeviationIntensity",
        "IntegratedIntensityEdge",
        "MeanIntensityEdge",
        "StdIntensityEdge",
        "MinIntensityEdge",
        "MaxIntensityEdge",
        "MaxZ",
        "MaxY",
        "MaxX",
        "CMI.X",
        "CMI.Y",
        "CMI.Z",
    }

    assert set(result) == {
        "object_id",
        "feature_name",
        "channel",
        "compartment",
        "value",
    }
    assert len(result["object_id"]) == EXPECTED_MEASUREMENT_COUNT
    assert set(result["feature_name"]) == expected_features
    assert result["channel"] == ["channel_a"] * EXPECTED_MEASUREMENT_COUNT
    assert result["compartment"] == ["nuclei"] * EXPECTED_MEASUREMENT_COUNT
    assert len(result["value"]) == EXPECTED_MEASUREMENT_COUNT


def test_compute_intensity_skips_empty_object_without_signal() -> None:
    image = np.zeros((2, 2, 2), dtype=float)
    labels = np.zeros((2, 2, 2), dtype=int)
    labels[0, 0, 0] = 1

    loader = SimpleNamespace(
        image=image,
        label_image=labels,
        object_ids=[1],
        channel="channel_b",
        compartment="cell",
    )

    result = compute_intensity(loader)

    assert result == {
        "object_id": [],
        "feature_name": [],
        "channel": [],
        "compartment": [],
        "value": [],
    }
