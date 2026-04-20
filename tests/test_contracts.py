"""Tests for data contract validation in zedprofiler.contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zedprofiler.contracts import (
    ExpectedValues,
    validate_column_name_schema,
    validate_image_array_shape_contracts,
    validate_image_array_type_contracts,
    validate_return_schema_contract,
)
from zedprofiler.exceptions import ContractError


@pytest.fixture
def expected_values_config_path(tmp_path: Path) -> Path:
    """Create a temporary expected-values TOML configuration file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[expected_values]
compartments = ["Nuclei", "Cytoplasm", "Cell", "Organoid"]
channels = ["DNA", "AGP", "ER", "Mito"]
""".strip()
    )
    return config_path


def test_validate_image_array_shape_contracts_accepts_valid_3d_array() -> None:
    arr = np.zeros((8, 16, 16), dtype=float)

    assert validate_image_array_shape_contracts(arr) is True


def test_validate_image_array_shape_contracts_accepts_single_channel_4d_array() -> None:
    arr = np.zeros((1, 8, 16, 16), dtype=float)

    assert validate_image_array_shape_contracts(arr) is True


def test_validate_image_array_shape_contracts_rejects_2d_array() -> None:
    arr = np.zeros((16, 16), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_shape_contracts_rejects_multichannel_4d_array() -> None:
    arr = np.zeros((2, 8, 16, 16), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_shape_contracts_rejects_multichannel_5d_array() -> None:
    arr = np.zeros((2, 2, 4, 8, 8), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_shape_contracts_rejects_all_singleton_3d_array() -> None:
    arr = np.zeros((1, 1, 1), dtype=float)

    with pytest.raises(ContractError):
        validate_image_array_shape_contracts(arr)


def test_validate_image_array_type_contracts_accepts_numeric_array() -> None:
    arr = np.zeros((8, 8, 8), dtype=np.float32)

    assert validate_image_array_type_contracts(arr) is True


def test_validate_image_array_type_contracts_rejects_non_numpy_array() -> None:
    arr = [[1, 2], [3, 4]]

    with pytest.raises(ContractError):
        validate_image_array_type_contracts(arr)  # type: ignore[arg-type]


def test_validate_image_array_type_contracts_rejects_non_numeric_dtype() -> None:
    arr = np.array([["a", "b"], ["c", "d"]], dtype=str)

    with pytest.raises(ContractError):
        validate_image_array_type_contracts(arr)


def test_expected_values_loads_config_and_adds_nochannel(
    expected_values_config_path: Path,
) -> None:
    values = ExpectedValues(expected_values_config_path)

    assert "Nuclei" in values.compartments
    assert "DNA" in values.channels
    assert "NoChannel" in values.channels
    assert "Intensity" in values.features


def test_expected_values_to_dict_returns_expected_keys(
    expected_values_config_path: Path,
) -> None:
    values = ExpectedValues(expected_values_config_path).to_dict()

    assert set(values.keys()) == {"compartments", "channels", "features"}


def test_validate_column_name_schema_accepts_valid_feature_column(
    expected_values_config_path: Path,
) -> None:
    valid_name = "Nuclei_DNA_Intensity_MeanIntensity"

    assert validate_column_name_schema(valid_name, expected_values_config_path) is True


def test_validate_column_name_schema_accepts_valid_metadata_column(
    expected_values_config_path: Path,
) -> None:
    valid_name = "Metadata_Storage_FilePath"

    assert validate_column_name_schema(valid_name, expected_values_config_path) is True


def test_validate_column_name_schema_rejects_non_string_column_name(
    expected_values_config_path: Path,
) -> None:
    with pytest.raises(ContractError):
        validate_column_name_schema(123, expected_values_config_path)  # type: ignore[arg-type]


def test_validate_column_name_schema_rejects_non_metadata_with_too_few_parts(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nuclei_DNA_Intensity"

    with pytest.raises(ContractError):
        validate_column_name_schema(invalid_name, expected_values_config_path)


def test_validate_column_name_schema_rejects_metadata_with_too_few_parts(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Metadata_Storage"

    with pytest.raises(ContractError):
        validate_column_name_schema(invalid_name, expected_values_config_path)


def test_validate_column_name_schema_rejects_unknown_compartment(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nucleus_DNA_Intensity_MeanIntensity"

    with pytest.raises(ContractError):
        validate_column_name_schema(invalid_name, expected_values_config_path)


def test_validate_column_name_schema_rejects_unknown_channel(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nuclei_GFP_Intensity_MeanIntensity"

    with pytest.raises(ContractError):
        validate_column_name_schema(invalid_name, expected_values_config_path)


def test_validate_column_name_schema_rejects_unknown_feature(
    expected_values_config_path: Path,
) -> None:
    invalid_name = "Nuclei_DNA_UnknownFeature_MeanIntensity"

    with pytest.raises(ContractError):
        validate_column_name_schema(invalid_name, expected_values_config_path)


def test_validate_return_schema_contract_accepts_valid_result() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    assert validate_return_schema_contract(result) is True


def test_validate_return_schema_contract_rejects_non_dict_result() -> None:
    with pytest.raises(ContractError):
        validate_return_schema_contract(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_return_schema_contract_rejects_missing_required_key() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)  # type: ignore[arg-type]


def test_validate_return_schema_contract_rejects_extra_key() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": {"Metadata_Object_ObjectID": 1},
        "extra": 123,
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_wrong_key_order() -> None:
    # Dict insertion order is deterministic in Python 3.7+; this order is intentional.
    result = {
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_invalid_image_array_type() -> None:
    result = {
        "image_array": [[1, 2], [3, 4]],
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_invalid_features_type() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": ["not", "a", "dict"],
        "metadata": {"Metadata_Object_ObjectID": 1},
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)


def test_validate_return_schema_contract_rejects_invalid_metadata_type() -> None:
    result = {
        "image_array": np.zeros((4, 8, 8), dtype=np.float32),
        "features": {"Nuclei_DNA_Intensity_MeanIntensity": 0.5},
        "metadata": ["not", "a", "dict"],
    }

    with pytest.raises(ContractError):
        validate_return_schema_contract(result)
