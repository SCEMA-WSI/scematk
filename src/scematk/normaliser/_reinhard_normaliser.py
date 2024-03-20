from ._normaliser import Normaliser
from ..process._processor import Processor
import dask.array as da
from dask.array import Array
from skimage import img_as_ubyte
from skimage.color import rgb2lab, lab2rgb

class ReinhardNormaliser(Normaliser):
    def __init__(self, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__("Reinhard Stain Normaliser", preprocessor, postprocessor)

    def fit(self, image: Array) -> None:
        assert isinstance(image, da.Array), "Image must be a dask array"
        assert image.ndim == 3, "Image must have 3 dimensions"
        assert image.shape[2] == 3, "Image must have 3 channels"
        image = self.preprocessor.process(image)
        image = da.map_blocks(rgb2lab, image, dtype="float32")
        self.means = da.mean(image, axis=(0, 1))
        self.stds = da.std(image, axis=(0, 1))
        self.fitted = True

    def run(self, image: Array) -> Array:
        assert self.fitted, "Normaliser must be fitted before running"
        assert isinstance(image, da.Array), "Image must be a dask array"
        assert image.ndim == 3, "Image must have 3 dimensions"
        assert image.shape[2] == 3, "Image must have 3 channels"
        image = self.preprocessor.process(image)
        image = da.map_blocks(rgb2lab, image, dtype="float32")
        means = da.mean(image, axis=(0, 1))
        stds = da.std(image, axis=(0, 1))
        image = (image - means) / stds * self.stds + self.means
        image = da.map_blocks(lambda x: img_as_ubyte(lab2rgb(x)), image, dtype="uint8")
        image = self.postprocessor.process(image)
        return image

    def fit_and_run(self, image: Array) -> Array:
        self.fit(image)
        return self.run(image)

    def _default_preprocessor(self) -> Processor:
        return Processor()

    def _default_postprocessor(self) -> Processor:
        return Processor()
