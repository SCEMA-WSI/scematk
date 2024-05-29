from ._mask import Mask
import dask.array as da
from dask.array import Array
from numpy import ndarray
from stardist import random_label_cmap
from typing import List

class LabelMask(Mask):
    def __init__(self, image: Array, info: dict, channel_names: List[str]) -> None:
        """Constructor for LabelMask

        Args:
            image (Array): The mask as a dask array
            info (dict): The info of the WSI as a dictionary
            channel_names (List[str]): The names of the channels in the mask
        """
        super().__init__(image, info, channel_names)
        assert image.dtype in [int, 'int32', 'int64'], "image must be an integer array"

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
        image = image.astype('float32')
        thumb = da.coarsen(da.mean, image, {0: coarsen_factor, 1: coarsen_factor}, trim_excess=True)
        return thumb.compute()