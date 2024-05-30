from typing import List

import dask.array as da
import numpy as np
from dask.array import Array
from numpy import ndarray

from ._image import Image


class UByteImage(Image):
    def __init__(self, image: Array, info: dict, channel_names: List[str]) -> None:
        """Constructor for UByteImage

        Args:
            image (Array): The WSI image as a dask array
            info (dict): The WSI metadata as a dictionary
            channel_names (List[str]): The names of the channels in the image
        """
        super().__init__(image, info, channel_names)
        assert self.dtype == "uint8", "image must be of type uint8"
        if self.ndim == 2:
            self.image = da.expand_dims(self.image, axis=-1)
        self.ndim = self.image.ndim
        self.shape = self.image.shape
        assert self.ndim == 3, "image must be 3D"

    def get_thumb(self, target_size: int = 512) -> ndarray:
        """Get a thumbnail of the image

        Args:
            target_size (int, optional): the target size of the largest dimension of the image. Defaults to 512.

        Raises:
            NotImplementedError: Only 1, 2, or 3 channels supported

        Returns:
            ndarray: The thumbnail image as a numpy array
        """
        assert isinstance(target_size, int), "target_size must be an integer"
        assert target_size > 0, "target_size must be greater than 0"
        coarsen_factor = max([s // target_size for s in self.shape])
        if coarsen_factor == 0:
            coarsen_factor = 1
        image = self.image
        thumb = da.coarsen(
            da.mean, image, {0: coarsen_factor, 1: coarsen_factor, 2: 1}, trim_excess=True
        )
        thumb = thumb.astype("uint8").compute()
        if self.shape[2] == 1:
            thumb = thumb.squeeze()
        elif self.shape[2] == 2:
            thumb = np.pad(thumb, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        elif self.shape[2] == 3:
            pass
        else:
            raise NotImplementedError("Only 1, 2, or 3 channels supported")
        return thumb
