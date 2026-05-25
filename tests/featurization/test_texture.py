from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from beartype import beartype
from pydantic import BaseModel, ConfigDict, field_validator

mahotas = pytest.importorskip("mahotas")

from zedprofiler.featurization.texture import compute_texture  # noqa: E402


class ImageSetLoaderModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_set_name: str = "texture"


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
def make_texture_image(
    shape: tuple[int, int, int], center: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros(shape, dtype=int)
    label = np.zeros(shape, dtype=int)
    z, y, x = center
    image[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 100
    label[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2] = 1
    return image, label


@pytest.mark.parametrize("shape,center", [((15, 15, 15), (7, 7, 7))])
def test_compute_texture_basic(
    shape: tuple[int, int, int], center: tuple[int, int, int]
) -> None:
    image, label = make_texture_image(shape, center)
    imgset = ImageSetLoaderModel()
    loader = ObjectLoaderModel(
        image=image, label_image=label, object_ids=[1], image_set_loader=imgset
    )

    df = compute_texture(loader, distance=1, grayscale=256)
    assert isinstance(df, pd.DataFrame)
    assert "Metadata_Object_ObjectID" in df.columns
