import sys
import types

import matplotlib
import numpy as np
import pandas as pd
import pytest

from zedprofiler.featurization import neighbors as neighbors_module

matplotlib.use("Agg", force=True)


EXPECTED_SHELLS_USED = 2
EXPECTED_OBJECT_COUNT = 6
EXPECTED_SECOND_OBJECT_ID = 2

if "image_analysis_3D" not in sys.modules:
    image_analysis_3D = types.ModuleType("image_analysis_3D")
    image_analysis_3D.__path__ = []  # type: ignore[attr-defined]
    sys.modules["image_analysis_3D"] = image_analysis_3D

if "image_analysis_3D.featurization_utils" not in sys.modules:
    featurization_utils = types.ModuleType("image_analysis_3D.featurization_utils")
    featurization_utils.__path__ = []  # type: ignore[attr-defined]
    sys.modules["image_analysis_3D.featurization_utils"] = featurization_utils

if "image_analysis_3D.featurization_utils.loading_classes" not in sys.modules:
    loading_classes = types.ModuleType(
        "image_analysis_3D.featurization_utils.loading_classes"
    )

    class ObjectLoader:
        pass

    loading_classes.ObjectLoader = ObjectLoader
    sys.modules["image_analysis_3D.featurization_utils.loading_classes"] = (
        loading_classes
    )


def test_neighbors_expand_box_clamps_to_global_bounds() -> None:
    result = neighbors_module.neighbors_expand_box(
        min_coor=0,
        max_coord=10,
        current_min=2,
        current_max=8,
        expand_by=3,
    )
    assert result == (0, 10)


def test_neighbors_expand_box_expands_without_clamping() -> None:
    result = neighbors_module.neighbors_expand_box(
        min_coor=0,
        max_coord=10,
        current_min=2,
        current_max=8,
        expand_by=1,
    )
    assert result == (1, 9)


def test_crop_3d_image_returns_expected_subvolume() -> None:
    image = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    cropped = neighbors_module.crop_3D_image(image=image, bbox=(1, 1, 1, 3, 4, 5))
    assert cropped.shape == (2, 3, 4)
    np.testing.assert_array_equal(cropped, image[1:3, 1:4, 1:5])


def test_compute_neighbors_counts_adjacent_and_distance_neighbors() -> None:
    label_image = np.zeros((1, 1, 4), dtype=int)
    label_image[0, 0, 0] = 1
    label_image[0, 0, 1] = 2
    label_image[0, 0, 3] = 3

    object_loader = types.SimpleNamespace(
        label_image=label_image,
        object_ids=[1, 2, 3],
    )

    result = neighbors_module.compute_neighbors(
        object_loader=object_loader,
        distance_threshold=1,
        anisotropy_factor=1,
    )

    assert result["object_id"] == [1, 2, 3]
    assert result["NeighborsCountAdjacent"] == [0, 0, 0]
    assert result["NeighborsCountByDistance-1"] == [1, 1, 0]


def test_get_coordinates_returns_centroids_for_selected_objects() -> None:
    nuclei_mask = np.zeros((2, 2, 2), dtype=int)
    nuclei_mask[0, 0, 0] = 1
    nuclei_mask[1, 1, 1] = 2

    coords = neighbors_module.get_coordinates(nuclei_mask, object_ids=[1, 2])

    assert list(coords.columns) == ["object_id", "x", "y", "z"]
    assert coords.shape == (2, 4)
    assert coords.loc[coords["object_id"] == 1, ["x", "y", "z"]].iloc[0].tolist() == [
        0.0,
        0.0,
        0.0,
    ]
    assert coords.loc[
        coords["object_id"] == EXPECTED_SECOND_OBJECT_ID,
        ["x", "y", "z"]].iloc[0].tolist() == [
        1.0,
        1.0,
        1.0,
    ]


def test_calculate_centroid_uses_column_mean() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    centroid = neighbors_module.calculate_centroid(coords)
    np.testing.assert_allclose(centroid, np.array([1.0, 2.0, 3.0]))


def test_euclidean_distance_from_centroid_matches_expected_values() -> None:
    coords = np.array([[1.0, 1.0, 1.0], [4.0, 5.0, 6.0]])
    centroid = np.array([1.0, 1.0, 1.0])

    distances = neighbors_module.euclidean_distance_from_centroid(coords, centroid)

    np.testing.assert_allclose(distances, np.array([0.0, np.sqrt(50.0)]))


def test_mahalanobis_distance_falls_back_to_euclidean_for_small_samples() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    centroid = np.array([0.0, 0.0, 0.0])

    mahalanobis = neighbors_module.mahalanobis_distance_from_centroid(coords, centroid)
    euclidean = neighbors_module.euclidean_distance_from_centroid(coords, centroid)

    np.testing.assert_allclose(mahalanobis, euclidean)


def test_mahalanobis_distance_uses_pseudo_inverse_for_singular_covariance() -> None:
    coords = np.zeros((20, 3))
    centroid = np.zeros(3)

    distances = neighbors_module.mahalanobis_distance_from_centroid(coords, centroid)

    np.testing.assert_allclose(distances, np.zeros(20))


def test_classify_cells_into_shells_handles_empty_input() -> None:
    results, centroid = neighbors_module.classify_cells_into_shells(
        coords={"object_id": [], "x": [], "y": [], "z": []}
    )

    assert centroid is None
    assert results == {
        "object_id": [],
        "ShellAssignments": [],
        "DistancesFromCenter": [],
        "DistancesFromExterior": [],
        "NormalizedDistancesFromCenter": [],
        "MaxShellsUsed": [],
    }


def test_classify_cells_into_shells_adjusts_shell_count_and_returns_results() -> None:
    coords = pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4, 5, 6],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    results, centroid = neighbors_module.classify_cells_into_shells(
        coords=coords,
        n_shells=5,
        method="euclidean",
        min_cells_per_shell=3,
    )

    assert centroid.shape == (3,)
    assert results["ShellsUsed"] == EXPECTED_SHELLS_USED
    assert len(results["object_id"]) == EXPECTED_OBJECT_COUNT
    assert len(results["ShellAssignments"]) == EXPECTED_OBJECT_COUNT
    assert len(results["DistancesFromCenter"]) == EXPECTED_OBJECT_COUNT
    assert len(results["DistancesFromExterior"]) == EXPECTED_OBJECT_COUNT
    assert len(results["NormalizedDistancesFromCenter"]) == EXPECTED_OBJECT_COUNT


def test_create_results_dataframe_builds_dataframe_from_results_dict() -> None:
    results = {
        "object_id": np.array([1, 2]),
        "ShellAssignments": np.array([0, 1]),
        "DistancesFromCenter": np.array([0.5, 1.5]),
        "DistancesFromExterior": np.array([1.0, 0.0]),
        "NormalizedDistancesFromCenter": np.array([0.25, 1.0]),
        "ShellsUsed": 2,
    }

    df = neighbors_module.create_results_dataframe(results)

    assert list(df.columns) == [
        "object_id",
        "ShellAssignments",
        "DistancesFromCenter",
        "DistancesFromExterior",
        "NormalizedDistancesFromCenter",
        "ShellsUsed",
    ]
    assert df.shape == (2, 6)


def test_create_results_dataframe_rejects_non_dict_input() -> None:
    with pytest.raises(ValueError, match="Input must be a results dictionary"):
        neighbors_module.create_results_dataframe([1, 2, 3])


def test_visualize_organoid_shells_returns_figure() -> None:
    coords = pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "z": [0.0, 1.0, 2.0],
        }
    )
    classification_results = {
        "ShellAssignments": np.array([0, 1, 1]),
        "ShellsUsed": 2,
    }

    fig = neighbors_module.visualize_organoid_shells(
        coords=coords,
        classification_results=classification_results,
        centroid=np.array([1.0, 1.0, 1.0]),
    )

    expected_axes = 2
    assert len(fig.axes) == expected_axes
    fig.canvas.draw()


def test_plot_distance_distributions_returns_figure() -> None:
    classification_results = {
        "ShellAssignments": np.array([0, 0, 1, 1]),
        "DistancesFromCenter": np.array([0.1, 0.2, 0.8, 0.9]),
        "DistancesFromExterior": np.array([0.9, 0.8, 0.2, 0.1]),
        "ShellsUsed": 2,
    }

    fig = neighbors_module.plot_distance_distributions(classification_results)

    expected_axes = 2
    assert len(fig.axes) == expected_axes
    fig.canvas.draw()
