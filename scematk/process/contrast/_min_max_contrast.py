import dask.array as da

from ...image._image import Image
from ...image._ubyte_image import UByteImage
from .._process import Process


class MinMaxContrast(Process):
    def __init__(self) -> None:
        """Constructor for the MinMaxContrast class."""
        super().__init__("Perform min-max contrast enhancement per channel")

    def run(self, image: Image) -> Image:
        """Performs min-max contrast enhancement on an image.

        Args:
            image (Image): The image to enhance the contrast of.

        Raises:
            ValueError: MinMaxContrast can only be applied to UByteImage

        Returns:
            Image: The image with enhanced contrast.
        """
        img = image.image
        if isinstance(image, UByteImage):
            img = img / 255
        minc = da.min(img, axis=(0, 1))
        img = img - minc
        maxc = da.max(img, axis=(0, 1))
        img = img / maxc
        if isinstance(image, UByteImage):
            img = img * 255
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise ValueError("MinMaxContrast can only be applied to UByteImage")
