from typing import List, Tuple

import dask.array as da
from dask.array import Array
from numpy import ndarray
from stardist import random_label_cmap

from ._mask import Mask


class LabelMask(Mask):
    def __init__(self, image: Array, info: dict, channel_names: List[str]) -> None:
        """Constructor for LabelMask

        Args:
            image (Array): The mask as a dask array
            info (dict): The info of the WSI as a dictionary
            channel_names (List[str]): The names of the channels in the mask
        """
        super().__init__(image, info, channel_names)
        assert image.dtype in [int, "int32", "int64"], "image must be an integer array"
        self.interpolation_strat = "nearest"

    def get_thumb(self, target_size: int = 512) -> ndarray:
        """Get a thumbnail of the mask

        Args:
            target_size (int, optional): The size of the thumbnail. Defaults to 512.

        Returns:
            ndarray: The thumbnail of the mask as a numpy array
        """
        assert isinstance(target_size, int), "target_size must be an integer"
        assert target_size > 0, "target_size must be greater than 0"
        coarsen_factor = max([s // target_size for s in self.shape])
        if coarsen_factor == 0:
            coarsen_factor = 1
        image = self.image
        image = image.astype("float32")
        thumb = da.coarsen(da.mean, image, {0: coarsen_factor, 1: coarsen_factor}, trim_excess=True)
        return thumb.compute()

    def _get_region_overlay(
        self,
        y_min: int,
        x_min: int,
        y_len: int,
        x_len: int,
        pad: bool = True,
        invert_overlay: bool = False,
    ) -> Tuple[ndarray | None, ndarray | None, str | None]:
        """Get a region of the mask as an overlay

        Args:
            y_min (int): The minimum y coordinate of the region
            x_min (int): The minimum x coordinate of the region
            y_len (int): The length of the region in the y direction
            x_len (int): The length of the region in the x direction
            pad (bool): Whether to pad the region around the edges
            invert_overlay (bool, optional): Whether to invert the overlay. Defaults to False.

        Returns:
            Tuple[ndarray | None, ndarray | None, str | None]: The mask region, the alpha channel, and the colormap
        """
        region = self.read_region(y_min, x_min, y_len, x_len, pad=pad)
        alpha_region = region > 0
        alpha_region = alpha_region.astype(float)
        cmap = random_label_cmap()
        if invert_overlay:
            return None, None, None
        return region, alpha_region, cmap
