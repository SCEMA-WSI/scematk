from typing import List

from dask.array import Array

from ._image import Image


class Mask(Image):
    def __init__(self, image: Array, info: dict, channel_names: List[str]) -> None:
        """Constructor for Mask

        Args:
            image (Array): The mask as a dask array
            info (dict): The info of the WSI as a dictionary
            channel_names (List[str]): The names of the channels in the mask
        """
        super().__init__(image, info, channel_names)
        assert self.ndim == 2, "mask must be 2D"
