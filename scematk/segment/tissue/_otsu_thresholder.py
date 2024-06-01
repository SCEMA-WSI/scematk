import dask.array as da
from skimage.filters import threshold_otsu

from ...image._binary_mask import BinaryMask
from ...image._image import Image
from ...image._ubyte_image import UByteImage
from ...process._process import Processor
from ...process.colour._rgb_to_grey import RGBToGrey
from ._tissue_segmenter import TissueSegmenter


class OtsuThresholder(TissueSegmenter):
    def __init__(
        self,
        preprocessor: Processor | None = None,
        postprocessor: Processor | None = None,
        below_thresh: bool = True,
    ):
        """Constructor for OtsuThresholder

        Args:
            preprocessor (Processor | None, optional): Preprocessor to run before segmentation. Defaults to None.
            postprocessor (Processor | None, optional): Postprocessor to run after segmentation. Defaults to None.
            below_thresh (bool, optional): Whether to segment below or above the threshold. Defaults to True.
        """
        super().__init__("Otsu Thresholder", preprocessor, postprocessor)
        assert isinstance(below_thresh, bool), "below_thresh must be a boolean"
        self.below_thresh = below_thresh

    def fit(self, image: Image) -> None:
        """Fit the model to the image

        Args:
            image (Image): Image to fit the model to

        Raises:
            NotImplementedError: OtsuThresholder only supports UByteImage at this time
        """
        image = self.preprocessor.run(image)
        if not isinstance(image, UByteImage):
            raise NotImplementedError("OtsuThresholder only supports UByteImage at this time")
        assert image.shape[2] == 1, "Image must be single channel"
        counts, _ = da.histogram(image.image, bins=256, range=[0, 256])
        self.threshold = threshold_otsu(hist=counts.compute(), nbins=256)
        self.fitted = True

    def run(self, image: Image) -> BinaryMask:
        """Run the model on the image

        Args:
            image (Image): Image to run the model on

        Raises:
            RuntimeError: Model must be fitted before running
            NotImplementedError: OtsuThresholder only supports UByteImage at this time

        Returns:
            BinaryMask: Segmented mask
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before running")
        image = self.preprocessor.run(image)
        if not isinstance(image, UByteImage):
            raise NotImplementedError("OtsuThresholder only supports UByteImage at this time")
        assert image.shape[2] == 1, "Image must be single channel"
        img = image.image
        img = da.squeeze(img, axis=2)
        if self.below_thresh:
            mask = img < self.threshold
        else:
            mask = img > self.threshold
        mask = BinaryMask(mask, image.info, ["TissueMask"])
        mask = self.postprocessor.run(mask)
        return mask

    def fit_and_run(self, image: Image) -> Image:
        """Fit the model to the image and run the model on the image

        Args:
            image (Image): Image to fit the model to and run the model on

        Returns:
            Image: Segmented mask
        """
        self.fit(image)
        return self.run(image)

    def _default_preprocessor(self) -> Processor:
        """Get the default preprocessor for the model

        Returns:
            Processor: Default preprocessor
        """
        proc = Processor()
        proc.add_process(RGBToGrey())
        return proc

    def _default_postprocessor(self) -> Processor:
        """Get the default postprocessor for the model

        Returns:
            Processor: Default postprocessor
        """
        return Processor()
