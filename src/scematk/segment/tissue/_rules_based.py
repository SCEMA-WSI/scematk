from .._segmenters import PrimarySegmenter
from ...process._processor import Processor
from ...process.colour._hsv_transform import RGBToHSV
import dask.array as da
from dask.array import Array

class RulesBasedTissueSegmenter(PrimarySegmenter):
    def __init__(self, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__("Rules Based Tissue Segmenter", preprocessor, postprocessor)
        self.fitted = True

    def fit(self, image: Array) -> None:
        pass

    def segment(self, image: Array) -> Array:
        image = self.preprocessor.process(image)
        image = RGBToHSV().process(image)
        saturation = image[:, :, 1]
        saturation = da.less_equal(saturation, 0.1)
        value = image[:, :, 2]
        value = da.greater_equal(value, 0.1)
        blank_mask = da.logical_and(saturation, value)
        hue = image[:, :, 0]
        hue1 = da.less(hue, 0.7)
        hue2 = da.greater(hue, 0.4)
        hue = da.logical_and(hue1, hue2)
        saturation = image[:, :, 1]
        saturation = da.greater(saturation, 0.1)
        hue_saturation = da.logical_and(hue, saturation)
        value = image[:, :, 2]
        value = da.less(value, 0.1)
        blue_mask = da.logical_or(hue_saturation, value)
        mask = da.logical_and(blank_mask, blue_mask)
        mask = da.logical_not(mask)
        mask = self.postprocessor.process(mask)
        return mask

    def fit_and_segment(self, image: Array) -> Array:
        return self.segment(image)

    def _default_preprocessor(self) -> Processor:
        return Processor()

    def _default_postprocessor(self) -> Processor:
        return Processor()
