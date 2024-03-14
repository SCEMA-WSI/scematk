import dask.array as da
from dask.array import Array
import matplotlib.pyplot as plt
import numpy as np

def show_thumb(image: Array, method: str = "mean", target_size: int = 256) -> None:
    assert isinstance(image, da.Array), f"Invalid image type: {type(image)}"
    assert len(image.shape) == 3 or len(image.shape) == 2, f"Invalid image shape: {image.shape}"
    assert method in ["median", "mean"], f"Invalid method: {method}"
    assert target_size > 0, f"Invalid target size: {target_size}"
    is_threed = len(image.shape) == 3
    if method == "median":
        method = da.median
    else:
        method = da.mean
    coarsen_factor = max(image.shape) // target_size
    if coarsen_factor == 0:
        coarsen_factor = 1
    if is_threed:
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

