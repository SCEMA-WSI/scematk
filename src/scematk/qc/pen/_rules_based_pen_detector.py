from ._pen_detector import PenDetector
from ...process._processor import Processor
from ...process.colour import RGBToHSV
import dask.array as da
from dask.array import Array

class RulesBasedPenDetector(PenDetector):
    def __init__(self, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__("Rules Based Pen Detector", preprocessor, postprocessor)
        self.fitted = True

    def fit(self, image: Array) -> None:
        pass

    def run(self, image: Array) -> Array:
        image = self.preprocessor.process(image)
        image = RGBToHSV().process(image)
        hue = image[:, :, 0]
        hue1 = da.less(hue, 0.7)
        hue2 = da.greater(hue, 0.4)
        hue = da.logical_and(hue1, hue2)
        saturation = image[:, :, 1]
        saturation = da.greater(saturation, 0.1)
        hue_saturation = da.logical_and(hue, saturation)
        value = image[:, :, 2]
        value = da.less(value, 0.1)
        mask = da.logical_or(hue_saturation, value)
        mask = self.postprocessor.process(mask)
        return mask

    def fit_and_run(self, image: Array) -> Array:
        return self.run(image)

    def _default_preprocessor(self) -> Processor:
        return Processor()

    def _default_postprocessor(self) -> Processor:
        return Processor()
