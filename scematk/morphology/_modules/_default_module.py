from typing import List, Optional

import pandas as pd
from dask.delayed import delayed
from numpy import ndarray
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border

from ._abc_module import _ABCModule


class DefaultModule(_ABCModule):
    def __init__(self, **kwargs):
        """Constructor for the default module."""
        super().__init__("Meta")

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
        """Measure the morphological features of an image tile

        Args:
            image (ndarray): An image tile
            mask (ndarray): A mask tile
            y_seed (int): The y-coordinate of the top left corner
            x_seed (int): The x-coordinate of the top left corner
            channel_names (List[str]): The names of the channel in the image
            mask_name (str): The name of the mask
            image_mpp (Optional[float]): The microns per pixel of the image. Defaults to None.

        Returns:
            pd.DataFrame: A pandas dataframe of the morphological features in the tile.
        """
        mask = clear_border(mask)
        data = regionprops_table(mask, properties=("label", "centroid"))
        data = pd.DataFrame(data)
        data["centroid-0"] += y_seed
        data["centroid-1"] += x_seed
        data.rename(
            columns={
                "label": "Meta_Global_Mask_Label",
                "centroid-0": f"Meta_{mask_name}_Mask_CentroidY",
                "centroid-1": f"Meta_{mask_name}_Mask_CentroidX",
            },
            inplace=True,
        )
        return data

    def _get_meta(self, channel_names: List[str], mask_name: str) -> dict:
        """Get the metadata of the dataframe

        Args:
            channel_names (List[str]): The names of the channels.
            mask_name (str): The name of the mask.

        Returns:
            dict: _description_
        """
        meta = {
            "Meta_Global_Mask_Label": int,
            f"Meta_{mask_name}_Mask_CentroidY": float,
            f"Meta_{mask_name}_Mask_CentroidX": float,
        }
        return meta
