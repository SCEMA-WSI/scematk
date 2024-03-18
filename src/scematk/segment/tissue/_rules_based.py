from .._segmenters import PrimarySegmenter
import dask.array as da
from dask.array import Array
from scematk.process._processor import Processor
from skimage.color import rgb2hsv

class RulesBasedTissueSegmenter(PrimarySegmenter):
    def __init__(self, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__("Rules Based Tissue Segmenter", preprocessor, postprocessor)
        self.fitted = True

    def fit(self, image: Array) -> None:
        pass

    def segment(self, image: Array) -> Array:
        image = self.preprocessor.process(image)
        image = da.map_blocks(rgb2hsv, image, dtype=float)
        saturation = image[:, :, 1]
        saturation = da.less_equal(saturation, 0.1)
        value = image[:, :, 2]
        value = da.greater_equal(value, 0.1)
        mask = da.logical_and(saturation, value)
        mask = self.postprocessor.process(mask)
        return mask

    def fit_and_segment(self, image: Array) -> Array:
        return self.segment(image)

    def _default_preprocessor(self) -> Processor:
        return Processor()

    def _default_postprocessor(self) -> Processor:
        return Processor()
