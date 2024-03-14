import dask.array as da
from dask.array import Array
from scematk.colour import rgb_to_grey
from skimage import img_as_ubyte

def grey_to_od(image: Array, clip = False) -> Array:
    assert image.ndim == 2, "Input image must be 2D"
    assert image.dtype in ["float32", "float64", "uint8"], "Input image must be float32, float64 or uint8"
    if image.dtype == "uint8":
        image = image / 255
    od_img = -da.log10(image)
    if clip:
        od_img = da.clip(od_img, 0, 1)
    return od_img

def gray_to_od(image: Array, clip = False) -> Array:
    return grey_to_od(image, clip)

def od_to_grey(image: Array, as_ubyte = False) -> Array:
    assert image.ndim == 2, "Input image must be 2D"
    assert image.dtype in ["float32", "float64"], "Input image must be float32 or float64"
    grey_img = da.exp(-image)
    if as_ubyte:
        grey_img = da.map_blocks(img_as_ubyte, grey_img)
    return grey_img

def od_to_gray(image: Array, as_ubyte = False) -> Array:
    return od_to_grey(image, as_ubyte)

def rgb_to_od(image: Array, clip = False) -> Array:
    image = rgb_to_grey(image)
    return grey_to_od(image, clip)