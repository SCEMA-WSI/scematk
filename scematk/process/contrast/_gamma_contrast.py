from ...image._image import Image
from ...image._ubyte_image import UByteImage
from .._process import Process


class GammaContrast(Process):
    def __init__(self, gamma: float | int = 1) -> None:
        """Constructor for the GammaContrast class.

        Args:
            gamma (float | int, optional): The gamma value to use for contrast adjustment. Defaults to 1.
        """
        super().__init__(f"Contrast adjustment using gamma correction of {str(gamma)}")
        assert isinstance(gamma, float) or isinstance(
            gamma, int
        ), "gamma must be convertable to float."
        self.gamma = gamma

    def run(self, image: Image) -> Image:
        """Adjusts the contrast of an image using gamma correction.

        Args:
            image (Image): The image to adjust the contrast of.

        Raises:
            NotImplementedError: Only UByteImage is supported at this time.
            NotImplementedError: Only UByteImage is supported at this time.

        Returns:
            Image: The image with adjusted contrast.
        """
        assert isinstance(image, Image), "image must be of type Image."
        if not isinstance(image, UByteImage):
            raise NotImplementedError(
                "GammaContrast only supports UByteImage at this time. Please raise an issue to change this."
            )
        img = image.image
        gamma = self.gamma
        if isinstance(image, UByteImage):
            img = img / 255.0
        img = img**gamma
        if isinstance(image, UByteImage):
            img = img * 255.0
            img = img.astype("uint8")
            return UByteImage(img, image.info, image.channel_names)
        else:
            raise NotImplementedError(
                "GammaContrast only supports UByteImage at this time. Please raise an issue to change this."
            )
