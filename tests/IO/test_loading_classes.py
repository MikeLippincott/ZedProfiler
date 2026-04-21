"""Tests for loading_classes module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zedprofiler.IO import loading_classes
from zedprofiler.IO.loading_classes import (
    ImageSetConfig,
    ImageSetLoader,
    ObjectLoader,
    TwoObjectLoader,
)

ZERO_LABEL = 0
ONE_LABEL = 1
TWO_LABEL = 2
EXPECTED_ANISOTROPY = 2.0
ORIGINAL_DNA_PIXEL = 10


class TestImageSetConfig:
    """Tests for ImageSetConfig dataclass."""

    def test_config_creation_defaults(self) -> None:
        """Test creating ImageSetConfig with defaults."""
        config = ImageSetConfig()
        assert config.image_set_name is None
        assert config.mask_key_name == []
        assert config.raw_image_key_name == []

    def test_config_creation_with_values(self) -> None:
        """Test creating ImageSetConfig with explicit values."""
        config = ImageSetConfig(
            image_set_name="test_set",
            mask_key_name=["mask1", "mask2"],
            raw_image_key_name=["raw1"],
        )
        assert config.image_set_name == "test_set"
        assert config.mask_key_name == ["mask1", "mask2"]
        assert config.raw_image_key_name == ["raw1"]

    def test_config_post_init_none_defaults(self) -> None:
        """Test that __post_init__ sets None fields to empty lists."""
        config = ImageSetConfig(image_set_name="test")
        assert config.mask_key_name == []
        assert config.raw_image_key_name == []


class TestImageSetLoaderMethods:
    """Tests for ImageSetLoader helper methods without filesystem coupling."""

    def test_retrieve_image_attributes_collects_only_mask_keys(self) -> None:
        """Mask-only unique object map is extracted from keys containing mask."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {
            "Nuclei_mask": np.array([[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]]),
            "DNA": np.array([[5, 6], [7, 8]]),
        }

        loader.retrieve_image_attributes()

        assert "Nuclei_mask" in loader.unique_mask_objects
        assert "DNA" not in loader.unique_mask_objects
        assert set(loader.unique_mask_objects["Nuclei_mask"].tolist()) == {
            ZERO_LABEL,
            ONE_LABEL,
            TWO_LABEL,
        }

    def test_get_compartments_and_image_names(self) -> None:
        """Compartment detection and non-compartment image naming are consistent."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {
            "Nuclei_mask": np.zeros((2, 2), dtype=np.int32),
            "Cell_mask": np.zeros((2, 2), dtype=np.int32),
            "DNA": np.ones((2, 2), dtype=np.int32),
        }

        compartments = loader.get_compartments()
        names = loader.get_image_names()

        assert compartments == ["Nuclei_mask", "Cell_mask"]
        assert names == ["DNA"]

    def test_get_unique_objects_in_compartments_filters_background(self) -> None:
        """Unique compartment objects should exclude background label 0."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {
            "Nuclei_mask": np.array(
                [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, ZERO_LABEL]],
                dtype=np.int32,
            ),
        }
        loader.compartments = ["Nuclei_mask"]

        loader.get_unique_objects_in_compartments()

        assert loader.unique_compartment_objects["Nuclei_mask"] == [
            ONE_LABEL,
            TWO_LABEL,
        ]

    def test_get_unique_objects_empty_compartments_raises_type_error(self) -> None:
        """Current behavior sets compartments to None then iterates and raises."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        loader.image_set_dict = {}
        loader.compartments = []

        with pytest.raises(TypeError):
            loader.get_unique_objects_in_compartments()

    def test_get_image_and_get_anisotropy(self) -> None:
        """Simple accessors return the expected image and anisotropy ratio."""
        loader = ImageSetLoader.__new__(ImageSetLoader)
        arr = np.arange(8).reshape((2, 2, 2))
        loader.image_set_dict = {"DNA": arr}
        loader.anisotropy_spacing = (2.0, 1.0, 1.0)

        assert np.array_equal(loader.get_image("DNA"), arr)
        assert loader.get_anisotropy() == EXPECTED_ANISOTROPY


class TestImageSetLoaderInit:
    """Tests that exercise ImageSetLoader __init__ with mocked reads."""

    def test_init_loads_channel_and_mask_images(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Initialization should load matching files and build derived attributes."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        (image_dir / "dna_raw.tif").touch()
        (image_dir / "ignore.txt").touch()
        (mask_dir / "nuc_mask.tif").touch()

        def _fake_imread(path: Path) -> np.ndarray:
            if "nuc_mask" in path.name:
                return np.array(
                    [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
                    dtype=np.int32,
                )
            return np.ones((2, 2), dtype=np.int32)

        monkeypatch.setattr(loading_classes.skimage.io, "imread", _fake_imread)

        loader = ImageSetLoader(
            image_set_path=image_dir,
            mask_set_path=mask_dir,
            anisotropy_spacing=(2.0, 1.0, 1.0),
            channel_mapping={"DNA": "dna_raw", "Nuclei_mask": "nuc_mask"},
            config=ImageSetConfig(
                image_set_name="set-01",
                mask_key_name=["mask"],
                raw_image_key_name=["raw"],
            ),
        )

        assert loader.image_set_name == "set-01"
        assert loader.anisotropy_factor == EXPECTED_ANISOTROPY
        assert set(loader.image_set_dict.keys()) == {"DNA", "Nuclei_mask"}
        assert loader.compartments == ["Nuclei_mask"]
        assert loader.image_names == ["DNA"]
        assert loader.unique_compartment_objects["Nuclei_mask"] == [
            ONE_LABEL,
            TWO_LABEL,
        ]

    def test_init_with_none_image_path_raises_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Current behavior for no images leads to TypeError in compartment pass."""
        monkeypatch.setattr(
            loading_classes.skimage.io,
            "imread",
            lambda _path: np.zeros((2, 2), dtype=np.int32),
        )

        with pytest.raises(TypeError):
            ImageSetLoader(
                image_set_path=None,
                mask_set_path=None,
                anisotropy_spacing=(1.0, 1.0, 1.0),
                channel_mapping={},
                config=ImageSetConfig(
                    mask_key_name=["mask"],
                    raw_image_key_name=["raw"],
                ),
            )


class TestObjectLoaders:
    """Tests for object-level loader classes."""

    def test_object_loader_drops_background_id(self) -> None:
        """ObjectLoader should omit the 0 label from object_ids."""
        label_image = np.array(
            [[ZERO_LABEL, ONE_LABEL], [TWO_LABEL, TWO_LABEL]],
            dtype=np.int32,
        )
        image = np.ones((2, 2), dtype=np.float32)
        image_set_loader = ImageSetLoader.__new__(ImageSetLoader)
        image_set_loader.image_set_dict = {
            "DNA": image,
            "Nuclei": label_image,
        }

        obj = ObjectLoader(
            image_set_loader=image_set_loader,
            channel_name="DNA",
            compartment_name="Nuclei",
        )

        assert obj.channel == "DNA"
        assert obj.compartment == "Nuclei"
        assert np.array_equal(obj.image, image)
        assert np.array_equal(obj.label_image, label_image)
        assert obj.object_ids == [ONE_LABEL, TWO_LABEL]

    def test_two_object_loader_copies_images_and_ids(self) -> None:
        """TwoObjectLoader should copy source arrays and preserve object IDs."""
        image_set_loader = ImageSetLoader.__new__(ImageSetLoader)
        image_set_loader.image_set_dict = {
            "Nuclei_mask": np.array([[ZERO_LABEL, ONE_LABEL]], dtype=np.int32),
            "DNA": np.array([[10, 11]], dtype=np.int32),
            "RNA": np.array([[20, 21]], dtype=np.int32),
        }
        image_set_loader.unique_compartment_objects = {"Nuclei_mask": [ONE_LABEL]}

        two = TwoObjectLoader(
            image_set_loader=image_set_loader,
            compartment="Nuclei_mask",
            channel1="DNA",
            channel2="RNA",
        )

        assert two.object_ids == [ONE_LABEL]
        assert np.array_equal(two.label_image, np.array([[ZERO_LABEL, ONE_LABEL]]))
        assert np.array_equal(two.image1, np.array([[10, 11]]))
        assert np.array_equal(two.image2, np.array([[20, 21]]))

        # Ensure they are copies, not views to the original arrays.
        two.image1[0, 0] = 999
        assert image_set_loader.image_set_dict["DNA"][0, 0] == ORIGINAL_DNA_PIXEL
