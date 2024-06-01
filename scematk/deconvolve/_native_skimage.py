import dask.array as da
from skimage import img_as_ubyte
from skimage.color import rgb2hed

from ..image._image import Image
from ..image._ubyte_image import UByteImage
from ..process._process import Processor
from ._stain_deconvolver import StainDeconvolver


class NativeSKImageStainDeconvolver(StainDeconvolver):
    def __init__(
        self,
        preprocessor: Processor | None = None,
        postprocessor: Processor | None = None,
        stain_type: str = "H&E",
    ) -> None:
        """Constructor for NativeSKImageStainDeconvolver

        Args:
            preprocessor (Processor, optional): Processor to run before deconvolution. Defaults to None.
            postprocessor (Processor, optional): Processor to run after deconvolution. Defaults to None.
            stain_type (str, optional): Type of stain to deconvolve. Defaults to "H&E".

        Raises:
            ValueError: If stain_type is not 'H&E' or 'H-DAB'
        """
        assert isinstance(stain_type, str), "stain_type must be a string"
        assert stain_type in ["H&E", "H-DAB"], "stain_type must be either 'H&E' or 'H-DAB'"
        if stain_type == "H&E":
            out_stains = ["Hematoxylin", "Eosin"]
        elif stain_type == "H-DAB":
            out_stains = ["Hematoxylin", "DAB"]
        else:
            raise ValueError("stain_type must be either 'H&E' or 'H-DAB'")
        super().__init__(
            "Stain deconvolver using scikit-image defaults", out_stains, preprocessor, postprocessor
        )
        self.fitted = True

    def fit(self, image: Image) -> None:
        """Fit the deconvolver to the image

        Args:
            image (UByteImage): Image to fit the deconvolver to
        """
        pass

    def run(self, image: Image) -> Image:
        """Run the deconvolution on the image

        Args:
            image (UByteImage): Image to run the deconvolution on

        Raises:
            ValueError: If image is not a UByteImage or if image does not have channels ['Red', 'Green', 'Blue']

        Returns:
            UByteImage: Deconvolved image
        """
        image = self.preprocessor.run(image)
        assert isinstance(image, UByteImage), "image must be a UByteImage"
        assert image.channel_names == [
            "Red",
            "Green",
            "Blue",
        ], "image must have channels ['Red', 'Green', 'Blue']"
        img = image.image
        img = da.map_blocks(rgb2hed, img, dtype="float32")
        img = da.clip(img, 0, 1)
        img = da.map_blocks(img_as_ubyte, img, dtype="uint8")
        if self.out_stains == ["Hematoxylin", "Eosin"]:
            img = img[..., [0, 1]]
        elif self.out_stains == ["Hematoxylin", "DAB"]:
            img = img[..., [0, 2]]
        else:
            raise ValueError(
                "out_stains must be either ['Hematoxylin', 'Eosin'] or ['Hematoxylin', 'DAB']"
            )
        deconv_image = UByteImage(img, image.info, self.out_stains)
        deconv_image_proc = self.postprocessor.run(deconv_image)
        return deconv_image_proc

    def fit_and_run(self, image: Image) -> Image:
        """Fit the deconvolver to the image and run the deconvolution

        Args:
            image (UByteImage): Image to fit the deconvolver to and run the deconvolution on

        Returns:
            UByteImage: Deconvolved image
        """
        return self.run(image)

    def _default_preprocessor(self) -> Processor:
        """Get the default preprocessor for the deconvolver

        Returns:
            Processor: Default preprocessor
        """
        return Processor()

    def _default_postprocessor(self) -> Processor:
        """Get the default postprocessor for the deconvolver

        Returns:
            Processor: Default postprocessor
        """
        return Processor()
