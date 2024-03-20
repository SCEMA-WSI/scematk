from .._process import Process
import dask.array as da
from dask.array import Array
from skimage.color import rgb2lab, lab2rgb
from skimage import img_as_ubyte

class LuminosityPercentileStandardiser(Process):
    def __init__(self, percentile: float = 0.95) -> None:
        super().__init__("Standardise Luminosity by Percentile")
        assert isinstance(percentile, float), "Percentile must be a float"
        assert 0 < percentile < 1, "Percentile must be between 0 and 1"
        self.percentile = percentile

    def process(self, image: Array) -> Array:
        image = da.map_blocks(rgb2lab, image, dtype=float)
        luminosity = image[:, :, 0]
        luminosity = da.percentile(luminosity.ravel(), self.percentile)
        image[:, :, 0] = da.clip(image[:, :, 0], 0, luminosity)
        image = da.map_blocks(lab2rgb, image, dtype=float)
        image = da.map_blocks(img_as_ubyte, image, dtype="uint8")
        return image
