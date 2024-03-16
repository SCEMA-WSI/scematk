from .._process import Process
import dask.array as da
from dask.array import Array
from skimage.color import rgb2hsv

class RGBToHSV(Process):
    def __init__(self) -> None:
        super().__init__("Convert an RGB image to an HSV image.")

    def process(self, image: Array) -> Array:
        assert isinstance(image, da.Array), f"Expected image to be dask.array.Array, got {type(image)}"
        assert image.ndim == 3, f"Expected image to be 3D, got {image.ndim}D"
        assert image.dtype == "uint8", f"Expected image to be uint8, got {image.dtype}"
        image = da.map_blocks(rgb2hsv, image, dtype="float64")
        return image