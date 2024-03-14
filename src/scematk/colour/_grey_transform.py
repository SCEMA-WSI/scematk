import dask.array as da
from dask.array import Array
from skimage import img_as_ubyte
from skimage.color import rgb2gray

def rgb_to_grey(image: Array, as_ubyte: bool = False) -> Array:
    assert image.shape[-1] == 3, "Input image must be RGB"
    assert image.dtype == "uint8", "Input image must be uint8"
    assert image.ndim == 3, "Input image must be 3D"
    image = da.map_blocks(rgb2gray, image, drop_axis=2, dtype="uint8")
    if as_ubyte:
        image = da.map_blocks(img_as_ubyte, image)
    return image

def rgb_to_gray(image: Array, as_ubyte: bool = False) -> Array:
    return rgb_to_grey(image, as_ubyte)