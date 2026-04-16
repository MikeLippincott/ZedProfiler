"""Tests for core data contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from zedprofiler.contracts import (
    MULTI_CHANNEL_CONTRACT,
    SINGLE_CHANNEL_CONTRACT,
    FloatArray,
    ImageArrayContract,
    IntArray,
)


def test_contract_constants_have_expected_values() -> None:
    """Single- and multi-channel constants should match documented shapes."""
    assert ImageArrayContract(dimensions=3, order="zyx") == SINGLE_CHANNEL_CONTRACT
    assert ImageArrayContract(dimensions=4, order="czyx") == MULTI_CHANNEL_CONTRACT


def test_image_array_contract_is_frozen() -> None:
    """Contracts are immutable to prevent accidental mutation in pipelines."""
    contract = ImageArrayContract(dimensions=3, order="zyx")

    with pytest.raises(FrozenInstanceError):
        contract.dimensions = 4  # type: ignore[misc]


def test_array_type_aliases_accept_numpy_ndarray_runtime_values() -> None:
    """Type aliases should correspond to ndarray values at runtime."""
    float_values: FloatArray = np.array([1.0, 2.0, 3.0], dtype=float)
    int_values: IntArray = np.array([1, 2, 3], dtype=int)

    assert isinstance(float_values, np.ndarray)
    assert isinstance(int_values, np.ndarray)
