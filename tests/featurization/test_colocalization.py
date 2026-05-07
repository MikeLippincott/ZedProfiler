import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

from zedprofiler.featurization import colocalization as coloc


def test_linear_costes_threshold_calculation_returns_finite_thresholds() -> None:
    x = np.linspace(0.01, 1.0, 200)
    y = 0.85 * x + 0.05 * np.sin(np.arange(x.size))
    t1, t2 = coloc.linear_costes_threshold_calculation(
        x, y, scale_max=255, fast_costes="Fast"
    )
    assert np.isfinite(t1)
    assert np.isfinite(t2)


def test_bisection_costes_threshold_calculation_returns_finite_thresholds() -> None:
    x = np.linspace(0.01, 1.0, 200)
    y = 0.9 * x + 0.03 * np.cos(np.arange(x.size))
    t1, t2 = coloc.bisection_costes_threshold_calculation(x, y, scale_max=255)
    assert np.isfinite(t1)
    assert np.isfinite(t2)


def test_prepare_two_images_for_colocalization_crops_expected_regions(
    monkeypatch: MonkeyPatch,
) -> None:
    label1 = np.zeros((4, 4, 4), dtype=int)
    label2 = np.zeros((4, 4, 4), dtype=int)
    label1[1:3, 1:3, 1:3] = 1
    label2[0:2, 0:2, 0:2] = 2

    img1 = np.arange(64).reshape(4, 4, 4)
    img2 = np.arange(100, 164).reshape(4, 4, 4)

    monkeypatch.setattr(coloc, "select_objects_from_label", lambda arr, _: arr)
    monkeypatch.setattr(coloc, "new_crop_border", lambda b1, b2, _img: (b1, b2))
    monkeypatch.setattr(
        coloc,
        "crop_3D_image",
        lambda img, bbox: img[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]],
    )

    out1, out2 = coloc.prepare_two_images_for_colocalization(
        label1, label2, img1, img2, object_id1=1, object_id2=2
    )

    assert out1.shape == (2, 2, 2)
    assert out2.shape == (2, 2, 2)
    np.testing.assert_array_equal(out1, img1[1:3, 1:3, 1:3])
    np.testing.assert_array_equal(out2, img2[0:2, 0:2, 0:2])


def test_compute_colocalization_identical_images_are_highly_colocalized() -> None:
    arr = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=float,
    )
    res = coloc.compute_colocalization(arr, arr, thr=0, fast_costes="Fast")

    expected_keys = {
        "Correlation",
        "MandersCoeffM1",
        "MandersCoeffM2",
        "OverlapCoeff",
        "MandersCoeffCostesM1",
        "MandersCoeffCostesM2",
        "RankWeightedColocalizationCoeff1",
        "RankWeightedColocalizationCoeff2",
    }
    assert expected_keys.issubset(res.keys())
    assert res["Correlation"] == pytest.approx(1.0, rel=1e-6)
    assert res["MandersCoeffM1"] == pytest.approx(1.0, rel=1e-6)
    assert res["MandersCoeffM2"] == pytest.approx(1.0, rel=1e-6)
    assert res["OverlapCoeff"] == pytest.approx(1.0, rel=1e-6)


def test_compute_colocalization_empty_combined_threshold_path() -> None:
    a = np.ones((2, 2, 2), dtype=float)
    b = np.ones((2, 2, 2), dtype=float) * 2.0

    res = coloc.compute_colocalization(a, b, thr=200, fast_costes="Fast")

    assert res["MandersCoeffM1"] == 0.0
    assert res["MandersCoeffM2"] == 0.0
    assert res["RankWeightedColocalizationCoeff1"] == 0.0
    assert res["RankWeightedColocalizationCoeff2"] == 0.0
    assert np.isnan(res["OverlapCoeff"])


def test_compute_colocalization_costes_dispatch(monkeypatch: MonkeyPatch) -> None:
    calls = {"bisection": 0, "linear": 0}

    def fake_bisection(
        _i1: np.ndarray, _i2: np.ndarray, _scale: int
    ) -> tuple[float, float]:
        calls["bisection"] += 1
        return 0.1, 0.1

    def fake_linear(
        _i1: np.ndarray,
        _i2: np.ndarray,
        _scale: int,
        _mode: str,
    ) -> tuple[float, float]:
        calls["linear"] += 1
        return 0.1, 0.1

    monkeypatch.setattr(coloc, "bisection_costes_threshold_calculation", fake_bisection)
    monkeypatch.setattr(coloc, "linear_costes_threshold_calculation", fake_linear)

    img1 = np.arange(1, 9, dtype=float).reshape(2, 2, 2)
    img2 = img1 + 1.0

    coloc.compute_colocalization(img1, img2, fast_costes="Accurate")
    coloc.compute_colocalization(img1, img2, fast_costes="Fast")

    assert calls["bisection"] == 1
    assert calls["linear"] == 1


def test_compute_colocalization_empty_input_raises() -> None:
    with pytest.raises(UnboundLocalError):
        coloc.compute_colocalization(np.array([]), np.array([]))
