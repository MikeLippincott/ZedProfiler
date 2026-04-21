from typing import Never

import numpy as np
from _pytest.monkeypatch import MonkeyPatch

from zedprofiler.featurization import texture


class DummyObjectLoader:
    def __init__(
        self,
        image: np.ndarray,
        label_image: np.ndarray,
        object_ids: np.ndarray,
    ) -> None:
        self.image = image
        self.label_image = label_image
        self.object_ids = object_ids


FEATURE_COUNT = 13
EXPECTED_DISTANCE = 2
FIRST_OBJECT_ID = 1
SECOND_OBJECT_ID = 2
THIRD_OBJECT_ID = 3
EXPECTED_OBJECT_COUNT = 2


def test_scale_image_constant_returns_zeros_uint8() -> None:
    image = np.full((2, 3, 4), 7, dtype=np.int16)

    out = texture.scale_image(image, num_gray_levels=256)

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.all(out == 0)


def test_scale_image_maps_min_max_to_requested_levels() -> None:
    image = np.array([0, 1023], dtype=np.int32)

    out = texture.scale_image(image, num_gray_levels=256)

    np.testing.assert_array_equal(out, np.array([0, 255], dtype=np.uint8))


def test_compute_texture_returns_expected_schema_and_lengths(
    monkeypatch: MonkeyPatch,
) -> None:
    image = np.arange(3 * 3 * 3, dtype=np.uint16).reshape((3, 3, 3))
    labels = np.zeros((3, 3, 3), dtype=np.int32)
    labels[0, 0, 0] = FIRST_OBJECT_ID
    labels[2, 2, 2] = SECOND_OBJECT_ID
    loader = DummyObjectLoader(
        image=image,
        label_image=labels,
        object_ids=np.array([FIRST_OBJECT_ID, SECOND_OBJECT_ID]),
    )

    fake_har = np.tile(np.arange(FEATURE_COUNT, dtype=float), (4, 1))

    def fake_haralick(
        *,
        ignore_zeros: bool,
        f: np.ndarray,
        distance: int,
        compute_14th_feature: bool,
    ) -> np.ndarray:
        assert ignore_zeros is True
        assert compute_14th_feature is False
        assert distance == EXPECTED_DISTANCE
        assert f.dtype == np.uint8
        return fake_har

    monkeypatch.setattr(texture.mahotas.features, "haralick", fake_haralick)

    out = texture.compute_texture(loader, distance=EXPECTED_DISTANCE, grayscale=64)

    assert set(out.keys()) == {"object_id", "texture_name", "texture_value"}
    assert len(out["object_id"]) == EXPECTED_OBJECT_COUNT * FEATURE_COUNT
    assert len(out["texture_name"]) == EXPECTED_OBJECT_COUNT * FEATURE_COUNT
    assert len(out["texture_value"]) == EXPECTED_OBJECT_COUNT * FEATURE_COUNT

    assert all(name.endswith("-64-2") for name in out["texture_name"])
    np.testing.assert_allclose(
        out["texture_value"][:FEATURE_COUNT],
        np.arange(FEATURE_COUNT, dtype=float),
    )
    np.testing.assert_allclose(
        out["texture_value"][FEATURE_COUNT:],
        np.arange(FEATURE_COUNT, dtype=float),
    )


def test_compute_texture_valueerror_from_haralick_yields_nan_values(
    monkeypatch: MonkeyPatch,
) -> None:
    image = np.ones((2, 2, 2), dtype=np.uint16)
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    labels[0, 0, 0] = THIRD_OBJECT_ID
    loader = DummyObjectLoader(
        image=image,
        label_image=labels,
        object_ids=np.array([THIRD_OBJECT_ID]),
    )

    def raise_value_error(**kwargs: object) -> Never:
        assert isinstance(kwargs, dict)
        raise ValueError("haralick failed")

    monkeypatch.setattr(texture.mahotas.features, "haralick", raise_value_error)

    out = texture.compute_texture(loader)

    assert len(out["object_id"]) == FEATURE_COUNT
    assert out["object_id"] == [THIRD_OBJECT_ID] * FEATURE_COUNT
    assert np.isnan(np.array(out["texture_value"], dtype=float)).all()


def test_compute_texture_skips_object_ids_not_present(
    monkeypatch: MonkeyPatch,
) -> None:
    image = np.arange(8, dtype=np.uint16).reshape((2, 2, 2))
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    labels[0, 0, 0] = FIRST_OBJECT_ID
    loader = DummyObjectLoader(
        image=image,
        label_image=labels,
        object_ids=np.array([FIRST_OBJECT_ID, 99]),
    )

    def fake_haralick_all_ones(**kwargs: object) -> np.ndarray:
        assert isinstance(kwargs, dict)
        return np.ones((4, FEATURE_COUNT), dtype=float)

    monkeypatch.setattr(
        texture.mahotas.features,
        "haralick",
        fake_haralick_all_ones,
    )

    out = texture.compute_texture(loader)

    assert len(out["object_id"]) == FEATURE_COUNT
    assert set(out["object_id"]) == {FIRST_OBJECT_ID}


def test_compute_texture_masks_non_object_voxels_inside_bbox(
    monkeypatch: MonkeyPatch,
) -> None:
    image = np.array([[[5, 9], [7, 11]]], dtype=np.uint16)  # shape (1, 2, 2)
    labels = np.array([[[1, 0], [0, 1]]], dtype=np.int32)  # same bbox, sparse object
    loader = DummyObjectLoader(
        image=image,
        label_image=labels,
        object_ids=np.array([FIRST_OBJECT_ID]),
    )

    seen = {}

    def fake_haralick(*, f: np.ndarray, **kwargs: object) -> np.ndarray:
        assert isinstance(kwargs, dict)
        seen["f"] = f.copy()
        return np.zeros((4, FEATURE_COUNT), dtype=float)

    monkeypatch.setattr(texture.mahotas.features, "haralick", fake_haralick)

    texture.compute_texture(loader, grayscale=256, distance=1)

    assert "f" in seen
    # Off-object voxels in the object's bbox should remain zero after masking/scaling
    assert seen["f"][0, 0, 1] == 0
    assert seen["f"][0, 1, 0] == 0
