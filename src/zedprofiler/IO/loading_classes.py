"""Data-loading classes for featurization workflows."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from types import SimpleNamespace

import numpy

try:
    import skimage.io as _skimage_io
except ImportError:
    _skimage_io = None

skimage = SimpleNamespace(io=SimpleNamespace(imread=None))
if _skimage_io is not None:
    skimage.io.imread = _skimage_io.imread

logging.basicConfig(level=logging.INFO)


def _read_image(path: pathlib.Path) -> numpy.ndarray:
    """Read an image with scikit-image when available."""
    if skimage.io.imread is None:
        raise ModuleNotFoundError(
            "scikit-image is required to load image files. "
            "Install `scikit-image` to use ImageSetLoader file I/O."
        )
    return skimage.io.imread(path)


@dataclasses.dataclass
class ImageSetConfig:
    """Configuration options for ImageSetLoader."""

    image_set_name: str | None = None
    mask_key_name: list[str] | None = None
    raw_image_key_name: list[str] | None = None

    # validate the arg types
    def __post_init__(self) -> None:
        """Initialize default values for None fields."""

        if not isinstance(self.image_set_name, (str, type(None))):
            raise TypeError("image_set_name must be a string or None")
        if not isinstance(self.mask_key_name, (list, type(None))):
            raise TypeError("mask_key_name must be a list of strings or None")
        if not isinstance(self.raw_image_key_name, (list, type(None))):
            raise TypeError("raw_image_key_name must be a list of strings or None")

        if self.mask_key_name is None:
            self.mask_key_name = []
        if self.raw_image_key_name is None:
            self.raw_image_key_name = []


class ImageSetLoader:
    """
    Load an image set consisting of raw z stack images and segmentation masks.

    A class to load an image set consisting of raw z stack images from multiple
    spectral channels and segmentation masks. The images are loaded into a
    dictionary, and various attributes and compartments are extracted from the
    images. The class also provides methods to retrieve images and their attributes.

    Parameters
    ----------
    image_set_path : pathlib.Path
        Path to the image set directory.
    mask_set_path : pathlib.Path
        Path to the mask set directory.
    anisotropy_spacing : tuple
        The anisotropy spacing of the images in format
        (z_spacing, y_spacing, x_spacing).
    channel_mapping : dict
        A dictionary mapping channel names to their corresponding image file names.
        Example: ``{'nuclei': 'nuclei_', 'cell': 'cell_', 'cytoplasm': 'cytoplasm_'}``

    Attributes
    ----------
    image_set_name : str
        The name of the image set.
    anisotropy_spacing : tuple
        The anisotropy spacing of the images.
    anisotropy_factor : float
        The anisotropy factor calculated from the spacing.
    image_set_dict : dict
        A dictionary containing the loaded images, with keys as channel names.
    unique_mask_objects : dict
        A dictionary containing unique object IDs for each mask in the image set.
    unique_compartment_objects : dict
        A dictionary containing unique object IDs for each compartment in the image set.
        A compartment is defined as a segmented region in the image (e.g., Cell,
        Cytoplasm, Nuclei, Organoid). The compartments are bounds for measurements.
    image_names : list
        A list of image names in the image set.
    compartments : list
        A list of compartment names in the image set.

    Methods
    -------
    retrieve_image_attributes()
        Retrieve unique object IDs for each mask in the image set.
    get_unique_objects_in_compartments()
        Retrieve unique object IDs for each compartment in the image set.
    get_image(key)
        Retrieve the image corresponding to the specified key.
    get_image_names()
        Retrieve the names of images in the image set.
    get_compartments()
        Retrieve the names of compartments in the image set.
    get_anisotropy()
        Retrieve the anisotropy factor.
    """

    def __init__(
        self,
        image_set_path: pathlib.Path,
        mask_set_path: pathlib.Path | None,
        anisotropy_spacing: tuple[float, float, float],
        channel_mapping: dict[str, str],
        config: ImageSetConfig | None = None,
    ) -> None:
        """Initialize the ImageSetLoader with paths, spacing, and mapping.

        Parameters
        ----------
        image_set_path : pathlib.Path
            Path to the image set directory.
        mask_set_path : pathlib.Path | None
            Path to the mask set directory.
        anisotropy_spacing : tuple
            The anisotropy spacing of the images. In format
            (z_spacing, y_spacing, x_spacing).
        channel_mapping : dict
            A dictionary mapping channel names to image file names.
        config : ImageSetConfig | None
            Optional configuration object with image_set_name, mask_key_name,
            and raw_image_key_name. If None, defaults are used.
        """
        if config is None:
            config = ImageSetConfig()

        channel_tokens = [str(value) for value in channel_mapping.values()]
        self.anisotropy_spacing = anisotropy_spacing
        self.anisotropy_factor = self.anisotropy_spacing[0] / self.anisotropy_spacing[1]
        self.image_set_name = config.image_set_name
        if image_set_path is None:
            channel_files = []
        else:
            channel_files = sorted(image_set_path.glob("*"))
            channel_files = [
                f
                for f in channel_files
                if f.suffix in [".tif", ".tiff"]
                and any(token in f.name for token in channel_tokens)
            ]

        self.mask_set_path = mask_set_path

        mask_files = sorted(mask_set_path.glob("*")) if mask_set_path else []
        mask_files = [
            f
            for f in mask_files
            if f.suffix in [".tif", ".tiff"]
            and any(token in f.name for token in channel_tokens)
        ]

        # Load images into a dictionary
        self.image_set_dict = {}
        for f in channel_files:
            for key, value in channel_mapping.items():
                if str(value) in f.name:
                    self.image_set_dict[key] = _read_image(f)
        for f in mask_files:
            for key, value in channel_mapping.items():
                if str(value) in f.name:
                    self.image_set_dict[key] = _read_image(f)

        self.retrieve_image_attributes()
        self.get_compartments()
        self.get_image_names()
        self.get_unique_objects_in_compartments()

    def retrieve_image_attributes(self) -> None:
        """
        This is also a quick and dirty way of loading two types of images:
            1. masks (multi-indexed segmentation masks)
            2. The spectral images to extract morphology features from

        My naming convention puts the work "mask" in the segmentation images this
        this is a way to differentiate each mask of each compartment
        apart from the spectral images.

        Future work should be to load the images in a more structured way
        that does not depend on the file naming convention.
        """
        self.unique_mask_objects = {}
        for key, value in self.image_set_dict.items():
            if "mask" in key:
                self.unique_mask_objects[key] = numpy.unique(value)

    def get_unique_objects_in_compartments(self) -> None:
        """Populate unique object IDs per compartment."""
        self.unique_compartment_objects = {}
        if len(self.compartments) == 0:
            self.compartments = None
        for compartment in self.compartments:
            self.unique_compartment_objects[compartment] = numpy.unique(
                self.image_set_dict[compartment]
            )
            # remove the 0 label
            self.unique_compartment_objects[compartment] = [
                x for x in self.unique_compartment_objects[compartment] if x != 0
            ]

    def get_image(self, key: str) -> numpy.ndarray:
        """Return an image array for a given key.

        Parameters
        ----------
        key : str
            Channel or mask key.

        Returns
        -------
        numpy.ndarray
            Image array for the requested key.
        """
        return self.image_set_dict[key]

    def get_image_names(self) -> list[str]:
        """Populate image (non-compartment) names.

        Returns
        -------
        list[str]
            List of image names excluding compartment masks.
        """
        compartments = (
            self.compartments
            if self.compartments is not None and isinstance(self.compartments, list)
            else []
        )
        self.image_names = [x for x in self.image_set_dict if x not in compartments]
        return self.image_names

    def get_compartments(self) -> list[str]:
        """Populate compartment names from available keys.

        Returns
        -------
        list[str]
            List of compartment keys.
        """
        self.compartments = [
            x
            for x in self.image_set_dict
            if "Nuclei" in x or "Cell" in x or "Cytoplasm" in x or "Organoid" in x
        ]
        return self.compartments

    def get_anisotropy(self) -> float:
        """Return the anisotropy factor for the image set.

        Returns
        -------
        float
            Ratio of z-spacing to y-spacing.
        """
        return self.anisotropy_spacing[0] / self.anisotropy_spacing[1]


class ObjectLoader:
    """
    A class to load objects from a labeled image and extract their properties.
    Where an object is defined as a segmented region in the image.
    This could be a cell, a nucleus, or any other compartment segmented.

    Parameters
    ----------
    image : numpy.ndarray
        The image from which to extract objects. Preferably a 3D image -> z, y, x
    label_image : numpy.ndarray
        The labeled image containing the segmented objects.
    channel_name : str
        The name of the channel from which the objects are extracted.
    compartment_name : str
        The name of the compartment from which the objects are extracted.

    Attributes
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    config : ImageSetConfig
        The configuration object containing image set parameters.

    Methods
    -------
    __init__(image, label_image, channel_name, compartment_name)
        Initializes the ObjectLoader with the image, label image, channel
        name, and compartment name.
    """

    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        channel_name: str,
        compartment_name: str,
    ) -> None:
        """Initialize object loader with image and labels.

        Parameters
        ----------
        image_set_loader : ImageSetLoader
            An instance of the ImageSetLoader class containing the image set.
        channel_name : str
            The name of the channel from which the objects are extracted.
        compartment_name : str
            The name of the compartment from which the objects are extracted.
        """

        self.channel = channel_name
        self.compartment = compartment_name
        self.image = (
            image_set_loader.image_set_dict[self.channel] if self.channel else None
        )
        self.label_image = (
            image_set_loader.image_set_dict[self.compartment]
            if self.compartment
            else None
        )
        # get the labeled image objects
        self.object_ids = numpy.unique(self.label_image)
        # drop the 0 label
        self.object_ids = [x for x in self.object_ids if x != 0]


class TwoObjectLoader:
    """
    A class to load two images and a label image for a specific compartment.
    This class is primarily used for loading images for two-channel
    analysis like co-localization.

    Parameters
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    compartment : str
        The name of the compartment for which the label image is loaded.
    channel1 : str
        The name of the first channel to be loaded.
    channel2 : str
        The name of the second channel to be loaded.

    Attributes
    ----------
    image_set_loader : ImageSetLoader
        An instance of the ImageSetLoader class containing the image set.
    compartment : str
        The name of the compartment for which the label image is loaded.
    label_image : numpy.ndarray
        The labeled image containing the segmented objects for the
        specified compartment.
    image1 : numpy.ndarray
        The image corresponding to the first channel.
    image2 : numpy.ndarray
        The image corresponding to the second channel.
    object_ids : numpy.ndarray
        The unique object IDs for the segmented objects in the specified compartment.

    Methods
    -------
    __init__(image_set_loader, compartment, channel1, channel2)
        Initializes the TwoObjectLoader with the image set loader,
        compartment, and channel names.
    """

    def __init__(
        self,
        image_set_loader: ImageSetLoader,
        compartment: str,
        channel1: str,
        channel2: str,
    ) -> None:
        """Initialize a two-channel loader for a compartment.

        Parameters
        ----------
        image_set_loader : ImageSetLoader
            Image set loader containing images and masks.
        compartment : str
            Compartment name for the label image.
        channel1 : str
            First channel name to load.
        channel2 : str
            Second channel name to load.
        """
        self.image_set_loader = image_set_loader
        self.compartment = compartment
        self.label_image = self.image_set_loader.image_set_dict[compartment].copy()
        self.image1 = self.image_set_loader.image_set_dict[channel1].copy()
        self.image2 = self.image_set_loader.image_set_dict[channel2].copy()
        self.object_ids = image_set_loader.unique_compartment_objects[compartment]
