from ._image import Image
from dask.array import Array
from typing import List

class Mask(Image):
    def __init__(self, image: Array, info: dict, channel_names: List[str]) -> None:
        super().__init__(image, info, channel_names)
        assert self.ndim == 2, "mask must be 2D"