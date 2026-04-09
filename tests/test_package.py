"""Baseline package tests for scaffolding."""

from __future__ import annotations

import zedprofiler
from zedprofiler import contracts
from zedprofiler._version import __version__ as expected_version
from zedprofiler.featurization import (
    areasizeshape,
    colocalization,
    granularity,
    intensity,
    neighbors,
    texture,
)


def test_version_is_exposed() -> None:
    assert zedprofiler.__version__ == expected_version


def test_input_contracts_are_defined() -> None:
    assert contracts.SINGLE_CHANNEL_CONTRACT.order == "zyx"
    assert contracts.MULTI_CHANNEL_CONTRACT.order == "czyx"


def test_public_module_imports_are_available() -> None:
    assert areasizeshape is not None
    assert colocalization is not None
    assert granularity is not None
    assert intensity is not None
    assert neighbors is not None
    assert texture is not None
