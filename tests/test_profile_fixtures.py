"""Test suite demonstrating the usage of test data profile fixtures.

This module shows how to use the available test data profiles in your tests.
"""

from __future__ import annotations

import numpy as np
from test_data_profiles import TestProfile

EXPECTED_IMAGE_NDIM = 3
EXPECTED_ALL_PROFILES_COUNT = 4
EXPECTED_FEATURE_TYPE_PROFILES_COUNT = 6


class TestProfileFixtures:
    """Tests demonstrating profile fixture usage."""

    def test_minimal_profile(self, minimal_profile: TestProfile) -> None:
        """Verify minimal profile has correct structure."""
        assert isinstance(minimal_profile.image_array, np.ndarray)
        assert minimal_profile.image_array.ndim == EXPECTED_IMAGE_NDIM
        assert len(minimal_profile.features) == 0
        assert len(minimal_profile.metadata) == 0

    def test_small_image_profile(self, small_image_profile: TestProfile) -> None:
        """Verify small profile has expected shape and data."""
        assert small_image_profile.image_array.shape == (4, 8, 8)
        assert "Nuclei_DNA_Intensity_MeanIntensity" in small_image_profile.features
        assert small_image_profile.metadata["Metadata_Object_ObjectID"] == 1

    def test_medium_image_profile(self, medium_image_profile: TestProfile) -> None:
        """Verify medium profile contains mixed feature types."""
        assert medium_image_profile.image_array.shape == (16, 32, 32)
        # Check for multiple feature types
        features = medium_image_profile.features
        assert any("Intensity" in f for f in features)
        assert any("Texture" in f for f in features)
        assert any("Areasizeshape" in f for f in features)

    def test_complete_profile(self, complete_profile: TestProfile) -> None:
        """Verify complete profile has comprehensive feature coverage."""
        assert complete_profile.image_array.shape == (24, 64, 64)
        features = complete_profile.features
        # Verify all feature types are present
        assert any("Intensity" in f for f in features)
        assert any("Texture" in f for f in features)
        assert any("Areasizeshape" in f for f in features)
        assert any("Colocalization" in f for f in features)
        assert any("Granularity" in f for f in features)
        assert any("Neighbors" in f for f in features)

    def test_intensity_profile(self, intensity_profile: TestProfile) -> None:
        """Verify intensity profile focuses on intensity features."""
        features = intensity_profile.features
        intensity_count = sum(1 for f in features if "Intensity" in f)
        total_count = len(features)
        assert intensity_count > total_count * 0.7  # Majority are intensity

    def test_texture_profile(self, texture_profile: TestProfile) -> None:
        """Verify texture profile focuses on texture features."""
        features = texture_profile.features
        texture_count = sum(1 for f in features if "Texture" in f)
        total_count = len(features)
        assert texture_count > total_count * 0.7  # Majority are texture

    def test_morphology_profile(self, morphology_profile: TestProfile) -> None:
        """Verify morphology profile focuses on shape/size features."""
        features = morphology_profile.features
        morph_count = sum(1 for f in features if "Areasizeshape" in f)
        total_count = len(features)
        assert morph_count > total_count * 0.7  # Majority are morphology

    def test_colocalization_profile(self, colocalization_profile: TestProfile) -> None:
        """Verify colocalization profile has colocalization features."""
        features = colocalization_profile.features
        assert all("Colocalization" in f for f in features)

    def test_granularity_profile(self, granularity_profile: TestProfile) -> None:
        """Verify granularity profile has granularity spectrum features."""
        features = granularity_profile.features
        assert all("Granularity" in f for f in features)

    def test_neighbors_profile(self, neighbors_profile: TestProfile) -> None:
        """Verify neighbors profile has neighbor features."""
        features = neighbors_profile.features
        assert all("Neighbors" in f for f in features)

    def test_all_profiles_collection(self, all_profiles: list[TestProfile]) -> None:
        """Verify collection fixture contains expected profiles."""
        assert len(all_profiles) == EXPECTED_ALL_PROFILES_COUNT
        assert all(isinstance(p, TestProfile) for p in all_profiles)
        # Verify increasing sizes
        sizes = [p.image_array.size for p in all_profiles]
        assert sizes == sorted(sizes)

    def test_all_feature_type_profiles(
        self, all_feature_type_profiles: list[TestProfile]
    ) -> None:
        """Verify feature type collection has all feature types."""
        assert len(all_feature_type_profiles) == EXPECTED_FEATURE_TYPE_PROFILES_COUNT
        assert all(isinstance(p, TestProfile) for p in all_feature_type_profiles)


class TestProfileWithVaryingSize:
    """Tests demonstrating parameterized profile fixtures."""

    def test_varying_profile_sizes(
        self, profile_with_varying_size: TestProfile
    ) -> None:
        """Verify profile_with_varying_size fixture produces valid profiles."""
        assert isinstance(profile_with_varying_size.image_array, np.ndarray)
        assert profile_with_varying_size.image_array.ndim == EXPECTED_IMAGE_NDIM
        assert (
            "Nuclei_DNA_Intensity_MeanIntensity" in profile_with_varying_size.features
        )
        assert profile_with_varying_size.metadata["Metadata_Object_ObjectID"] == 1


def test_profile_to_dict() -> None:
    """Verify TestProfile.to_dict() method works correctly."""
    profile = TestProfile(
        image_array=np.ones((4, 8, 8)),
        features={"test_feature": 0.5},
        metadata={"test_meta": "value"},
    )

    profile_dict = profile.to_dict()
    assert isinstance(profile_dict, dict)
    assert "image_array" in profile_dict
    assert "features" in profile_dict
    assert "metadata" in profile_dict
    assert np.array_equal(profile_dict["image_array"], profile.image_array)
