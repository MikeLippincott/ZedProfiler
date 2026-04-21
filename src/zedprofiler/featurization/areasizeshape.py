"""Area, size, and shape features for 3D objects."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol

import numpy as np

from zedprofiler.exceptions import ZedProfilerError


class SupportsImageSetLoader(Protocol):
    """Minimal image-set loader interface required by this module."""

    anisotropy_spacing: tuple[float, float, float]


class SupportsObjectLoader(Protocol):
    """Minimal object loader interface required by this module."""

    label_image: np.ndarray
    object_ids: Sequence[int]


def _empty_feature_result() -> dict[str, list[float]]:
    """Return deterministic empty output schema for area/size/shape features."""
    return {
        "object_id": [],
        "Volume": [],
        "CenterX": [],
        "CenterY": [],
        "CenterZ": [],
        "BboxVolume": [],
        "MinX": [],
        "MaxX": [],
        "MinY": [],
        "MaxY": [],
        "MinZ": [],
        "MaxZ": [],
        "Extent": [],
        "EulerNumber": [],
        "EquivalentDiameter": [],
        "SurfaceArea": [],
    }


def compute(
    image_set_loader: SupportsImageSetLoader | None = None,
    object_loader: SupportsObjectLoader | None = None,
) -> dict[str, list[float]]:
    """Compute area/size/shape features for one object loader.

    This supports two invocation modes:
    - no arguments: returns an empty deterministic schema so dispatchers can
      call the function without crashing.
    - both loaders provided: executes feature extraction.
    """
    if image_set_loader is None and object_loader is None:
        return _empty_feature_result()
    if image_set_loader is None or object_loader is None:
        raise ZedProfilerError(
            "areasizeshape.compute requires both image_set_loader and "
            "object_loader for execution."
        )

    return measure_3D_area_size_shape(
        image_set_loader=image_set_loader,
        object_loader=object_loader,
    )


def _get_skimage_measure() -> object:
    """Return `skimage.measure` or raise a user-facing dependency error."""
    try:
        return import_module("skimage.measure")
    except ImportError as exc:
        raise ZedProfilerError(
            "areasizeshape requires scikit-image for area/size/shape computation."
        ) from exc


def calculate_surface_area(
    label_object: np.ndarray,
    props: dict[str, np.ndarray],
    spacing: tuple[float, float, float],
) -> float:
    """Calculate surface area for one labeled object using marching cubes."""
    measure = _get_skimage_measure()

    volume = label_object[
        max(props["bbox-0"][0], 0) : min(props["bbox-3"][0], label_object.shape[0]),
        max(props["bbox-1"][0], 0) : min(props["bbox-4"][0], label_object.shape[1]),
        max(props["bbox-2"][0], 0) : min(props["bbox-5"][0], label_object.shape[2]),
    ]
    volume_truths = volume > 0
    verts, faces, _normals, _values = measure.marching_cubes(
        volume_truths,
        method="lewiner",
        spacing=spacing,
        level=0,
    )
    return measure.mesh_surface_area(verts, faces)


def measure_3D_area_size_shape(
    image_set_loader: SupportsImageSetLoader,
    object_loader: SupportsObjectLoader,
) -> dict[str, list[float]]:
    """Measure area/size/shape features for each non-zero label object."""
    measure = _get_skimage_measure()

    label_object = object_loader.label_image
    spacing = image_set_loader.anisotropy_spacing
    unique_objects = object_loader.object_ids

    features_to_record = _empty_feature_result()

    desired_properties = [
        "area",
        "bbox",
        "centroid",
        "bbox_area",
        "extent",
        "euler_number",
        "equivalent_diameter",
    ]
    for label in unique_objects:
        if label == 0:
            continue
        subset_lab_object = label_object.copy()
        subset_lab_object[subset_lab_object != label] = 0
        props = measure.regionprops_table(
            subset_lab_object,
            properties=desired_properties,
        )

        features_to_record["object_id"].append(label)
        features_to_record["Volume"].append(props["area"].item())
        features_to_record["CenterX"].append(props["centroid-2"].item())
        features_to_record["CenterY"].append(props["centroid-1"].item())
        features_to_record["CenterZ"].append(props["centroid-0"].item())
        features_to_record["BboxVolume"].append(props["bbox_area"].item())
        features_to_record["MinX"].append(props["bbox-2"].item())
        features_to_record["MaxX"].append(props["bbox-5"].item())
        features_to_record["MinY"].append(props["bbox-1"].item())
        features_to_record["MaxY"].append(props["bbox-4"].item())
        features_to_record["MinZ"].append(props["bbox-0"].item())
        features_to_record["MaxZ"].append(props["bbox-3"].item())
        features_to_record["Extent"].append(props["extent"].item())
        features_to_record["EulerNumber"].append(props["euler_number"].item())
        features_to_record["EquivalentDiameter"].append(
            props["equivalent_diameter"].item()
        )

        try:
            features_to_record["SurfaceArea"].append(
                calculate_surface_area(
                    label_object=label_object,
                    props=props,
                    spacing=spacing,
                )
            )
        except (RuntimeError, ValueError):
            features_to_record["SurfaceArea"].append(np.nan)

    return features_to_record
