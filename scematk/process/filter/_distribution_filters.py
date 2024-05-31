import numpy as np
from dask_image.ndfilters import (
    maximum_filter,
    median_filter,
    minimum_filter,
    percentile_filter,
)
from skimage.morphology import disk

from ...image._image import Image
from ...image._ubyte_image import UByteImage
from ...process._process import Process


class MaximumFilter(Process):
    def __init__(self, radius: float, metric: str = "micron") -> None:
        """Constructor for MaximumFilter

        Args:
            radius (float): Radius of the filter
            metric (str, optional): Units of the radius. Defaults to 'micron'.
        """
        assert isinstance(radius, (int, float)), "radius must be a number"
        assert radius > 0, "radius must be positive"
        assert isinstance(metric, str), "metric must be a string"
        assert metric in ["micron", "pixel"], 'metric must be either "micron" or "pixel"'
        self.radius = radius
        self.metric = metric
        super().__init__(name=f"Maximum Filter with a radius of {radius} {metric}s")

    def run(self, image: Image) -> Image:
        """Run the MaximumFilter process

        Args:
            image (Image): SCEMATK Image object

        Raises:
            NotImplementedError: MaximumFilter only supports UByteImage objects at this time

        Returns:
            Image: Image object with the MaximumFilter applied
        """
        assert isinstance(image, Image), "image must be an Image"
        radius = image.pixel_from_micron(self.radius) if self.metric == "micron" else self.radius
        footprint = disk(radius)
        footprint = np.expand_dims(footprint, axis=-1)
        img = image.image
        img = maximum_filter(img, footprint=footprint)
        if isinstance(image, UByteImage):
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise NotImplementedError("MaximumFilter only supports UByteImage objects at this time")


class MedianFilter(Process):
    def __init__(self, radius: float, metric: str = "micron") -> None:
        """Constructor for MedianFilter

        Args:
            radius (float): Radius of the filter
            metric (str, optional): Units of the radius. Defaults to 'micron'.
        """
        assert isinstance(radius, (int, float)), "radius must be a number"
        assert radius > 0, "radius must be positive"
        assert isinstance(metric, str), "metric must be a string"
        assert metric in ["micron", "pixel"], 'metric must be either "micron" or "pixel"'
        self.radius = radius
        self.metric = metric
        super().__init__(name=f"Median Filter with a radius of {radius} {metric}s")

    def run(self, image: Image) -> Image:
        """Run the MedianFilter process

        Args:
            image (Image): SCEMATK Image object

        Raises:
            NotImplementedError: MedianFilter only supports UByteImage objects at this time

        Returns:
            Image: Image object with the MedianFilter applied
        """
        assert isinstance(image, Image), "image must be an Image"
        radius = image.pixel_from_micron(self.radius) if self.metric == "micron" else self.radius
        footprint = disk(radius)
        footprint = np.expand_dims(footprint, axis=-1)
        img = image.image
        img = median_filter(img, footprint=footprint)
        if isinstance(image, UByteImage):
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise NotImplementedError("MedianFilter only supports UByteImage objects at this time")


class MinimumFilter(Process):
    def __init__(self, radius: float, metric: str = "micron") -> None:
        """Constructor for MinimumFilter

        Args:
            radius (float): Radius of the filter
            metric (str, optional): Units of the radius. Defaults to 'micron'.
        """
        assert isinstance(radius, (int, float)), "radius must be a number"
        assert radius > 0, "radius must be positive"
        assert isinstance(metric, str), "metric must be a string"
        assert metric in ["micron", "pixel"], 'metric must be either "micron" or "pixel"'
        self.radius = radius
        self.metric = metric
        super().__init__(name=f"Minimum Filter with a radius of {radius} {metric}s")

    def run(self, image: Image) -> Image:
        """Run the MinimumFilter process

        Args:
            image (Image): SCEMATK Image object

        Raises:
            NotImplementedError: MinimumFilter only supports UByteImage objects at this time

        Returns:
            Image: Image object with the MinimumFilter applied
        """
        assert isinstance(image, Image), "image must be an Image"
        radius = image.pixel_from_micron(self.radius) if self.metric == "micron" else self.radius
        footprint = disk(radius)
        footprint = np.expand_dims(footprint, axis=-1)
        img = image.image
        img = minimum_filter(img, footprint=footprint)
        if isinstance(image, UByteImage):
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise NotImplementedError("MinimumFilter only supports UByteImage objects at this time")


class PercentileFilter(Process):
    def __init__(self, radius: float, percentile: float, metric: str = "micron") -> None:
        """Constructor for PercentileFilter

        Args:
            radius (float): Radius of the filter
            percentile (float): Percentile value to select from local neighbourhood
            metric (str, optional): Units of the radius. Defaults to 'micron'.
        """
        assert isinstance(radius, (int, float)), "radius must be a number"
        assert radius > 0, "radius must be positive"
        assert isinstance(percentile, (int, float)), "percentile must be a number"
        assert 0 <= percentile <= 100, "percentile must be between 0 and 100"
        assert isinstance(metric, str), "metric must be a string"
        assert metric in ["micron", "pixel"], 'metric must be either "micron" or "pixel"'
        self.radius = radius
        self.percentile = percentile
        self.metric = metric
        super().__init__(
            name=f"Percentile Filter with a radius of {radius} {metric}s and a percentile of {percentile}"
        )

    def run(self, image: Image) -> Image:
        """Run the PercentileFilter process

        Args:
            image (Image): SCEMATK Image object

        Raises:
            NotImplementedError: PercentileFilter only supports UByteImage objects at this time

        Returns:
            Image: Image object with the PercentileFilter applied
        """
        assert isinstance(image, Image), "image must be an Image"
        radius = image.pixel_from_micron(self.radius) if self.metric == "micron" else self.radius
        footprint = disk(radius)
        footprint = np.expand_dims(footprint, axis=-1)
        img = image.image
        img = percentile_filter(img, footprint=footprint, percentile=self.percentile)
        if isinstance(image, UByteImage):
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise NotImplementedError(
                "PercentileFilter only supports UByteImage objects at this time"
            )
