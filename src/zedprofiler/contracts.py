"""Core data contracts shared across featurizers.

The package accepts:
- Single-channel 3D arrays shaped (z, y, x)
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import tomli

from zedprofiler.exceptions import ContractError

EXPECTED_SPATIAL_DIMS = 3
TWO_DIMENSIONAL = 2
FOUR_DIMENSIONAL = 4
FIVE_OR_MORE_DIMENSIONS = 5
REQUIRED_RETURN_KEYS = ("image_array", "features", "metadata")


def validate_image_array_shape_contracts(
    arr: np.ndarray,
) -> bool:
    """
    Validate the input array for dimensionality

    Parameters
    ----------
    arr : np.ndarray
        Input array to validate

    Returns
    -------
    bool
        The status of the validation

    Raises
    ------
    ContractError
        If the input array does not meet the expected contract
    """

    arr_shape = arr.shape
    if len(arr_shape) == TWO_DIMENSIONAL:
        raise ContractError(
            f"Input array has shape {arr_shape} with {TWO_DIMENSIONAL} dimensions. "
            f"Expected {EXPECTED_SPATIAL_DIMS} dimensions."
        )
    elif len(arr_shape) == FOUR_DIMENSIONAL and arr_shape[0] > 1:
        raise ContractError(
            f"Input array has shape {arr_shape} with {FOUR_DIMENSIONAL} dimensions, "
            "but the first dimension (channels) has size "
            f"{arr_shape[0]}. Expected a single-channel 3D array."
        )
    elif (
        len(arr_shape) >= FIVE_OR_MORE_DIMENSIONS
        and arr_shape[0] > 1
        and arr_shape[1] > 1
    ):
        raise ContractError(
            f"Input array has shape {arr_shape} with {len(arr_shape)} dimensions. "
            f"Expected {EXPECTED_SPATIAL_DIMS} dimensions."
        )

    for dim_size in arr_shape:
        if dim_size <= 0:
            raise ContractError(
                f"Input array has shape {arr_shape} with non-positive dimension size. "
                "All dimensions must have size greater than 0."
            )
    if sum(arr_shape) == len(arr_shape):
        raise ContractError(
            f"Input array has shape {arr_shape} with one or more dimensions of size 1. "
            "Expected all three dimensions to have size greater than 1."
        )
    return True


def validate_image_array_type_contracts(
    arr: np.ndarray,
) -> bool:
    """
    Validate the input array for type

    Parameters
    ----------
    arr : np.ndarray
        Input array to validate

    Returns
    -------
    bool
        The status of the validation

    Raises
    ------
    ContractError
        If the input array does not meet the expected contract
    """
    if not isinstance(arr, np.ndarray):
        raise ContractError(f"Input is of type {type(arr)}, expected a numpy array.")
    # check for numeric dtype (int or float) in the array
    if not np.issubdtype(arr.dtype, np.number):
        raise ContractError(
            f"Input array has dtype {arr.dtype}, expected a numeric dtype "
            "(int or float)."
        )
    return True


def validate_return_schema_contract(
    result: dict[str, object],
) -> bool:
    """Validate return schema keys, types, and deterministic key ordering."""
    if not isinstance(result, dict):
        raise ContractError(f"Return result must be a dict, got {type(result)}.")

    actual_keys = tuple(result.keys())
    if actual_keys != REQUIRED_RETURN_KEYS:
        raise ContractError(
            "Return result keys must match required deterministic order "
            f"{REQUIRED_RETURN_KEYS}, got {actual_keys}."
        )

    if not isinstance(result["image_array"], np.ndarray):
        raise ContractError("Return result key 'image_array' must be a numpy array.")
    if not isinstance(result["features"], dict):
        raise ContractError("Return result key 'features' must be a dict.")
    if not isinstance(result["metadata"], dict):
        raise ContractError("Return result key 'metadata' must be a dict.")

    return True


@dataclass
class ExpectedValues:
    """Expected values for feature naming validation tests."""

    config_file_path: pathlib.Path
    compartments: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Load expected values from a TOML configuration file."""
        config = tomli.loads(self.config_file_path.read_text())
        self.compartments = list(set(config["expected_values"]["compartments"]))
        self.channels = list(set(config["expected_values"]["channels"]))
        # add "NoChannel" as a valid channel for metadata columns
        # This is automatically added in the ZedProfiler
        # regardless of input channel we want this added
        # Add "NoChannel" as a valid channel for metadata columns.
        self.channels.append("NoChannel")
        self.features = [
            "AreaSizeShape",
            "Correlation",
            "Granularity",
            "Intensity",
            "Neighbors",
            "Texture",
            "SAMMed3D",
            "CHAMMI-75",
        ]

    def to_dict(self) -> dict[str, list[str]]:
        """Return expected values as a dictionary."""
        return {
            "compartments": self.compartments,
            "channels": self.channels,
            "features": self.features,
        }


def validate_column_name_schema(
    column_name: str,
    expected_values_config_path: pathlib.Path,
) -> bool:
    """
    Validate the column name schema for required fields and types

    Parameters
    ----------
    column_name : str
        The column name to validate
    expected_values_config_path : pathlib.Path
        Path to the configuration file containing expected values for validation
    Returns
    -------
    bool
        The status of the validation
    Raises
    ------
    ContractError
        If the column name does not meet the expected schema
    """
    non_metadata_underscore_seperated_parts = 4
    metadata_underscore_seperated_parts = 3

    expected_values = ExpectedValues(expected_values_config_path).to_dict()
    # check if the column name is a string
    if not isinstance(column_name, str):
        raise ContractError(f"Column name must be a string, got {type(column_name)}")

    # check if the column name has at least 4 parts separated by underscores
    parts = column_name.split("_")
    if (
        len(parts) < non_metadata_underscore_seperated_parts
        and "Metadata" not in column_name
    ):
        msg = (
            "Column name must have at least "
            f"{non_metadata_underscore_seperated_parts} "
            "parts separated by underscores, "
            f"got {len(parts)} parts in '{column_name}'"
        )
        raise ContractError(msg)

    if "Metadata" in column_name:
        if len(parts) < metadata_underscore_seperated_parts:
            raise ContractError(
                "Metadata column name must have at least "
                f"{metadata_underscore_seperated_parts} "
                "parts separated by "
                f"underscores, got {len(parts)} parts in '{column_name}'"
            )
        return True

    compartment = parts[0]
    channel = parts[1]
    feature = parts[2]

    # check if the compartment is one of the expected values
    expected_compartments = expected_values.get("compartments", [])
    expected_channels = expected_values.get("channels", [])
    expected_features = expected_values.get("features", [])
    msg = (
        f"Compartment '{compartment}' is not in the expected values: "
        f"{expected_compartments}"
    )
    if compartment not in expected_compartments:
        raise ContractError(msg)
    msg = f"Channel '{channel}' is not in the expected values: {expected_channels}"
    if channel not in expected_channels:
        raise ContractError(msg)
    msg = f"Feature '{feature}' is not in expected values: {expected_features}"
    if feature not in expected_features:
        raise ContractError(msg)
    return True
