from .._process import Process
import dask.array as da
from dask.array import Array
from skimage import img_as_ubyte
from skimage.color import rgb2gray

class GreyToOD(Process):
    def __init__(self) -> None:
        super().__init__("Convert a greyscale image to an optical density image.")

    def process(self, image: Array) -> Array:
        assert isinstance(image, da.Array), f"Expected image to be dask.array.Array, got {type(image)}"
        assert image.ndim == 2, f"Expected image to be 2D, got {image.ndim}D"
        assert image.dtype == "uint8", f"Expected image to be uint8, got {image.dtype}"
        image = -da.log10(image / 255)
        image = da.clip(image, 0, 1) * 255
        image = image.astype("uint8")
        return image

class RGBToOD(Process):
    def __init__(self) -> None:
        super().__init__("Convert an RGB image to an optical density image.")

    def process(self, image: Array) -> Array:
        assert isinstance(image, da.Array), f"Expected image to be dask.array.Array, got {type(image)}"
        assert image.ndim == 3, f"Expected image to be 3D, got {image.ndim}D"
        assert image.dtype == "uint8", f"Expected image to be uint8, got {image.dtype}"
        image = da.map_blocks(rgb2gray, image, dtype="float64")
        image = da.map_blocks(img_as_ubyte, image, dtype="uint8")
        image = -da.log10(image / 255)
        image = da.clip(image, 0, 1) * 255
        image = image.astype("uint8")
        return image