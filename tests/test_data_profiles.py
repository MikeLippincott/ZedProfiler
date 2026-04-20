"""Test data profiles utilities for ZedProfiler feature extraction.

This module provides the TestProfile dataclass for working with test data.
All fixture definitions are in conftest.py for easy pytest discovery and reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TestProfile:
    """A complete test profile with image data, features, and metadata."""

    image_array: np.ndarray
    features: dict
    metadata: dict

    def to_dict(self) -> dict:
        """Convert profile to dictionary format."""
        return {
            "image_array": self.image_array,
            "features": self.features,
            "metadata": self.metadata,
        }
