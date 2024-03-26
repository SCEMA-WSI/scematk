import dask.array as da
from dask.array import Array
import matplotlib.pyplot as plt
import numpy as np

def show_thumb(image: Array, method: str = "mean", target_size: int = 256) -> None:
    assert isinstance(image, Array), "image should be a dask array"
    assert image.ndim == 2 or image.ndim == 3, "image should be a 2D or 3D array"
    assert isinstance(method, str), "method should be a string"
    assert method in ["mean", "median"], "method should be either 'mean' or 'median'"
    assert isinstance(target_size, int), "target_size should be an integer"
    is_three_d = image.ndim == 3
    if method == "mean":
        method = da.mean
    elif method == "median":
        method = da.median
    else:
        raise ValueError("method should be either 'mean' or 'median'")
    coarsen_factor = max(image.shape[:2]) // target_size
    if coarsen_factor == 0:
        coarsen_factor = 1
    if is_three_d:
        thumb = da.coarsen(method, image, {0: coarsen_factor, 1: coarsen_factor, 2: 1}, trim_excess=True).compute()
        if np.max(thumb) > 1:
            thumb = thumb.astype(np.uint8)
        plt.imshow(thumb)
    else:
        thumb = da.coarsen(method, image, {0: coarsen_factor, 1: coarsen_factor}, trim_excess=True).compute()
        if np.max(thumb) > 1:
            thumb = thumb.astype(np.uint8)
        plt.imshow(thumb, cmap="gray")
    plt.axis("off")
    plt.show()
