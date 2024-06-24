from abc import ABC, abstractmethod
from typing import List, Optional

import dask.dataframe as dd
import pandas as pd
from dask.delayed import delayed
from numpy import ndarray


class _ABCModule(ABC):
    def __init__(self, name: str) -> None:
        """Constructor for the _ABCModule class

        Args:
            name (str): The name of the module
        """
        assert isinstance(name, str), "Name should be a string"
        self.name = name

    @delayed
    def _measure(
        self,
        image: ndarray,
        mask: ndarray,
        y_seed: int,
        x_seed: int,
        channel_names: List[str],
        mask_name: str,
        image_mpp: Optional[float] = None,
    ) -> pd.DataFrame:
        """Measure the features of the image

        Args:
            image (ndarray): The image to measure
            mask (ndarray): The mask to measure
            y_seed (int): The y coordinate of the seed
            x_seed (int): The x coordinate of the seed
            channel_names (List[str]): The names of the channels
            mask_name (str): The name of the mask
            image_mpp (Optional[float], optional): The microns per pixel of the image. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe containing the measurements
        """
        pass

    @abstractmethod
    def _get_meta(self, channel_names: List[str], mask_name: str) -> dict:
        """Get the column names and types of the measurements

        Args:
            channel_names (List[str]): The names of the channels
            mask_name (str): The name of the mask

        Returns:
            dict: A dictionary containing the column names and types
        """
        pass

    def measure(
        self,
        image: ndarray,
        mask: ndarray,
        block_size: int,
        overlap_size: int,
        y_index: int,
        x_index: int,
        channel_names: List[str],
        mask_name: str,
        image_mpp: Optional[float] = None,
    ) -> dd.DataFrame:
        """Measure the features of the image

        Args:
            image (ndarray): The image to measure
            mask (ndarray): The mask to measure
            block_size (int): The size of the block
            overlap_size (int): The size of the overlap
            y_index (int): The y index of the tile
            x_index (int): The x index of the tile
            channel_names (List[str]): The names of the channels
            mask_name (str): The name of the mask
            image_mpp (Optional[float], optional): The microns per pixel of the image. Defaults to None.

        Returns:
            dd.DataFrame: A dask dataframe containing the measurements
        """
        y_seed = max(0, y_index * block_size - overlap_size)
        x_seed = max(0, x_index * block_size - overlap_size)
        return dd.from_delayed(
            self._measure(image, mask, y_seed, x_seed, channel_names, mask_name, image_mpp),
            meta=self._get_meta(channel_names, mask_name),
        )
