from typing import Optional

import dask.array as da
from skimage.segmentation import expand_labels

from ...image._image import Image
from ...image._label_mask import LabelMask
from ...process._process import Processor
from ._secondary_segmenter import SecondarySegmenter


class ExpansionSegmenter(SecondarySegmenter):
    def __init__(
        self,
        distance: float,
        units: str = "micron",
        preprocessor: Optional[Processor] = None,
        postprocessor: Optional[Processor] = None,
    ) -> None:
        """Expansion segmenter that expands labels in an image.

        Args:
            distance (float, optional): The maximum distance to expand the labels. Defaults to None.
            units (str, optional): The units of the distance. Must be either 'micron' or 'pixel'. Defaults to 'micron'.
            preprocessor (Processor, optional): The preprocessor to apply to the image before running the segmenter. Defaults to None.
            postprocessor (Processor, optional): The postprocessor to apply to the image after running the segmenter. Defaults to None.
        """
        assert distance is not None, "Distance must be provided"
        assert isinstance(distance, (int, float)), "Distance must be a number"
        assert isinstance(units, str), "Units must be a string"
        assert units in ["micron", "pixel"], "Units must be either 'micron' or 'pixel'"
        self.distance = distance
        self.units = units
        super().__init__(
            f"Expansion segmenter with a maximum distance of {distance} {units}s",
            preprocessor,
            postprocessor,
        )
        self.fitted = True

    def fit(self, image: Image) -> None:
        """Fit the segmenter to the image.

        Args:
            image (Image): Image to fit the segmenter to.
        """
        pass

    def run(self, image: Image) -> Image:
        """Run the segmenter on the image.

        Args:
            image (Image): Image to run the segmenter on.

        Raises:
            ValueError: Image must be an instance of LabelMask

        Returns:
            Image: Image with expanded labels.
        """
        assert isinstance(image, LabelMask), "Image must be an instance of LabelMask"
        image = self.preprocessor.run(image)
        distance = self.distance
        units = self.units
        if units == "micron":
            if image.mpp:
                distance = distance / image.mpp
            else:
                raise ValueError(
                    "Image must have a microns per pixel value to convert distance to pixels"
                )
        img = image.image
        depth = int(2 * distance + 1)
        img = da.map_overlap(
            lambda x: expand_labels(x, distance=distance), img, depth=depth, dtype=img.dtype
        )
        expanded_image = LabelMask(img, image.info, channel_names=["Mask"])
        return self.postprocessor.run(expanded_image)

    def fit_and_run(self, image: Image) -> Image:
        """Fit the segmenter to the image and run the segmenter on the image.

        Args:
            image (LabelMask): Image to fit the segmenter to and run the segmenter on.

        Returns:
            LabelMask: Image with expanded labels.
        """
        return self.run(image)

    def _default_preprocessor(self) -> Processor:
        """Create a default preprocessor for the segmenter.

        Returns:
            Processor: Default preprocessor.
        """
        return Processor()

    def _default_postprocessor(self) -> Processor:
        """Create a default postprocessor for the segmenter.

        Returns:
            Processor: Default postprocessor.
        """
        return Processor()
