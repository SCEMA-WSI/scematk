from dask_image.ndmorph import binary_closing
from skimage.morphology import disk

from ...image._binary_mask import BinaryMask
from ...image._image import Image
from ...process._process import Process


class BinaryClosing(Process):
    def __init__(self, radius: float, metric: str = "micron") -> None:
        """Constructor for BinaryClosing

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
        super().__init__(name=f"Binary Closing with a radius of {radius} {metric}s")

    def run(self, image: Image) -> Image:
        """Run the BinaryClosing process

        Args:
            image (BinaryMask): SCEMATK BinaryMask object

        Returns:
            BinaryMask: BinaryMask object with the BinaryClosing applied
        """
        assert isinstance(image, BinaryMask), "image must be a BinaryMask"
        radius = image.pixel_from_micron(self.radius) if self.metric == "micron" else self.radius
        mask = image.image
        footprint = disk(radius)
        mask = binary_closing(mask, structure=footprint)
        return BinaryMask(mask, image.info, image.channel_names)
