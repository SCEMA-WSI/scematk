from .._segmenters import PrimarySegmenter
from ...process._processor import Processor
from ...process.colour import RGBToOD
import dask.array as da
from dask.array import Array
from skimage.filters import threshold_otsu

class GlobalOtsuTissueSegmenter(PrimarySegmenter):
    def __init__(self, preprocessor: Processor = None, postprocessor: Processor = None, invert_foreground: bool = False) -> None:
        super().__init__("Global Otsu Thresholder Tissue Segmenter", preprocessor, postprocessor)
        assert isinstance(invert_foreground, bool), "Invert foreground must be a boolean"
        self.invert_foreground = invert_foreground

    def fit(self, image: Array) -> None:
        image = self.preprocessor.process(image)
        assert isinstance(image, da.Array), "Image must be a dask array"
        assert image.ndim == 2, "Image must be 2D"
        assert image.dtype == "uint8", "Image must be uint8"
        counts, _ = da.histogram(image, bins=256, range=[0, 256])
        self.threshold = threshold_otsu(hist=counts.compute(), nbins=256)
        self.fitted = True

    def segment(self, image: Array) -> Array:
        assert self.fitted, "Segmenter must be fitted before segmenting"
        image = self.preprocessor.process(image)
        assert isinstance(image, da.Array), "Image must be a dask array"
        assert image.ndim == 2, "Image must be 2D"
        assert image.dtype == "uint8", "Image must be uint8"
        threshold = self.threshold
        if self.invert_foreground:
            image = da.less(image, threshold)
        else:
            image = da.greater_equal(image, threshold)
        return self.postprocessor.process(image)

    def fit_and_segment(self, image: Array) -> Array:
        self.fit(image)
        return self.segment(image)

    def _default_preprocessor(self) -> Processor:
        return Processor().add_process(RGBToOD())

    def _default_postprocessor(self) -> Processor:
        return Processor()