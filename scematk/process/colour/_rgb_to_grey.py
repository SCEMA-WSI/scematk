import dask.array as da
from skimage import img_as_ubyte
from skimage.color import rgb2gray

from ...image._image import Image
from ...image._ubyte_image import UByteImage
from .._process import Process


class RGBToGrey(Process):
    def __init__(self) -> None:
        """Constructor for the RGBToGrey class."""
        super().__init__("Convert an RGB UByte Image to a Greyscale UByte Image")

    def run(self, image: Image) -> Image:
        """Converts an RGB UByte Image to a Greyscale UByte Image.

        Args:
            image (Image): The RGB UByte Image to convert to greyscale.

        Returns:
            Image: The greyscale UByte Image.
        """
        assert isinstance(image, UByteImage), "Image must be a UByteImage"
        assert image.ndim == 3, "Image must have 3 dimensions"
        assert image.shape[2] == 3, "Image must have 3 channels"
        assert image.channel_names == [
            "Red",
            "Green",
            "Blue",
        ], "Image must have the channel names 'Red', 'Green', 'Blue'"
        img = image.image
        grey_img = da.map_blocks(
            lambda x: img_as_ubyte(rgb2gray(x)), img, drop_axis=2, dtype="uint8"
        )
        grey_img = da.expand_dims(grey_img, axis=2)
        return UByteImage(grey_img, image.info, ["Grey"])
