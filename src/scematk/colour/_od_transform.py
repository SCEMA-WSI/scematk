import dask.array as da
from dask.array import Array
from scematk.colour import rgb_to_grey

def grey_to_od(image: Array) -> Array:
    assert image.ndim == 2, "Input image must be 2D"
    assert image.dtype == "uint8", "Input image must be uint8"
    od_img = -da.log10(image / 255)
    od_img = da.clip(od_img, 0, 1)
    od_img = da.round(od_img * 255, 0)
    od_img = od_img.astype("uint8")
    return od_img

def gray_to_od(image: Array) -> Array:
    return grey_to_od(image)

def rgb_to_od(image: Array) -> Array:
    image = rgb_to_grey(image)
    return grey_to_od(image)