from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

from zedprofiler.featurization.intensity import compute_intensity


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "intensity"


class ObjectLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: np.ndarray
    label_image: np.ndarray
    object_ids: list[int]
    image_set_loader: ImageSetLoaderModel
    compartment: str = "Cell"
    channel: str = "Ch1"

    @field_validator("image", "label_image", mode="before")
    @classmethod
    def to_array(_cls, v: object) -> np.ndarray:
        return np.asarray(v)


@beartype
def make_label_and_image(
    shape: tuple[int, int, int], center: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros(shape, dtype=float)
    label = np.zeros(shape, dtype=int)
    z, y, x = center
    image[z, y, x] = 50.0
    label[z, y, x] = 1
    return image, label


@pytest.mark.parametrize("shape,center", [((6, 6, 6), (3, 3, 3))])
def test_compute_intensity_basic(
    shape: tuple[int, int, int], center: tuple[int, int, int]
) -> None:
    img, lab = make_label_and_image(shape, center)
    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=img,
        label_image=lab,
        object_ids=[1],
        image_set_loader=imgset,
    )

    df = compute_intensity(loader)
    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns
