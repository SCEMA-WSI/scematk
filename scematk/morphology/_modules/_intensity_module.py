from typing import List, Optional, Type, Union

import numpy as np
import pandas as pd
from dask.delayed import delayed
from numpy import ndarray
from skimage.measure import regionprops_table
from skimage.morphology import remove_small_holes
from skimage.segmentation import clear_border, find_boundaries

from ._abc_module import _ABCModule


def intensity_std(mask: ndarray, img: ndarray) -> float:
    """Get the standard deviation of stain intensity in an image region

    Args:
        mask (ndarray): Mask of region
        img (ndarray): Image

    Returns:
        float: The standard deviation of the intensity in the mask
    """
    return np.std(img[mask == 1])


def intensity_median(mask: ndarray, img: ndarray) -> float:
    """Get the median of stain intensity in an image region

    Args:
        mask (ndarray): Mask of region
        img (ndarray): Image

    Returns:
        float: The median of the intensity in the mask
    """
    return np.median(img[mask == 1])


def get_border(mask: ndarray) -> ndarray:
    """Get the border of a mask

    Args:
        mask (ndarray): A mask of a region

    Returns:
        ndarray: A mask of the border region
    """
    area = np.prod(mask.shape)
    mask = remove_small_holes(mask, area)
    border = find_boundaries(mask, connectivity=2, mode="inner")
    border[[0, -1], :] = mask[[0, -1], :]
    border[:, [0, -1]] = mask[:, [0, -1]]
    return border


def border_mean(mask: ndarray, img: ndarray) -> float:
    """Get the mean intensity in the border of a region

    Args:
        mask (ndarray): Mask of image region
        img (ndarray): Image

    Returns:
        float: The mean intensity in the border of a region
    """
    border = get_border(mask)
    return np.mean(img[border])


def border_min(mask: ndarray, img: ndarray) -> float:
    """Get the minimum intensity in the border of a region

    Args:
        mask (ndarray): Mask of image region
        img (ndarray): Image

    Returns:
        float: The minimum intensity in the border of a region
    """
    border = get_border(mask)
    return np.min(img[border])


def border_max(mask: ndarray, img: ndarray) -> float:
    """Get the maximum intensity in the border of a region

    Args:
        mask (ndarray): Mask of image region
        img (ndarray): Image

    Returns:
        float: The maximum intensity in the border of a region
    """
    border = get_border(mask)
    return np.max(img[border])


def border_std(mask: ndarray, img: ndarray) -> float:
    """Get the standard deviation of intensity in the border of a region

    Args:
        mask (ndarray): Mask of image region
        img (ndarray): Image

    Returns:
        float: The standard deviation of intensity in the border of a region
    """
    border = get_border(mask)
    return np.std(img[border])


def border_median(mask: ndarray, img: ndarray) -> float:
    """Get the median intensity in the border of a region

    Args:
        mask (ndarray): Mask of image region
        img (ndarray): Image

    Returns:
        float: The median intensity in the border of a region
    """
    border = get_border(mask)
    return np.median(img[border])


class IntensityModule(_ABCModule):
    def __init__(self, **kwargs):
        """Constructor of intensity module"""
        super().__init__("Intensity")

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
        """Measure the intensity features of objects in an image tile

        Args:
            image (ndarray): The image to analyse
            mask (ndarray): The mask to analyse
            y_seed (int): The y-coordinate of the top left corner of the image
            x_seed (int): The x-coordinate of the top left corner of the image
            channel_names (List[str]): The names of the channels of the image
            mask_name (str): The name of the mask
            image_mpp (Optional[float], optional): the microns per pixel of the image. Defaults to None.

        Returns:
            pd.DataFrame: A dataframe of the intensity morphological measurements of the object in the image
        """
        mask = clear_border(mask)
        data = regionprops_table(
            mask,
            intensity_image=image,
            properties=["label", "intensity_max", "intensity_mean", "intensity_min"],
            extra_properties=[
                intensity_std,
                intensity_median,
                border_mean,
                border_min,
                border_max,
                border_std,
                border_median,
            ],
        )
        data = pd.DataFrame(data)
        rename_dict = {"label": "Meta_Global_Mask_Label"}
        for i, channel_name in enumerate(channel_names):
            rename_dict[f"intensity_max-{i}"] = f"Intensity_{mask_name}_{channel_name}_MaxIntensity"
            rename_dict[f"intensity_mean-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_MeanIntensity"
            )
            rename_dict[f"intensity_min-{i}"] = f"Intensity_{mask_name}_{channel_name}_MinIntensity"
            rename_dict[f"intensity_std-{i}"] = f"Intensity_{mask_name}_{channel_name}_StdIntensity"
            rename_dict[f"intensity_median-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_MedianIntensity"
            )
            rename_dict[f"border_mean-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_BorderMeanIntensity"
            )
            rename_dict[f"border_min-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_BorderMinIntensity"
            )
            rename_dict[f"border_max-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_BorderMaxIntensity"
            )
            rename_dict[f"border_std-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_BorderStdIntensity"
            )
            rename_dict[f"border_median-{i}"] = (
                f"Intensity_{mask_name}_{channel_name}_BorderMedianIntensity"
            )
        data.rename(columns=rename_dict, inplace=True)
        return data

    def _get_meta(self, channel_names: List[str], mask_name: str) -> dict:
        """Get the column metadata of the dataframe

        Args:
            channel_names (List[str]): The channel names
            mask_name (str): The mask name

        Returns:
            dict: A dictionary of the column metadata
        """
        meta: dict[str, Union[Type[int], Type[float]]] = {
            "Meta_Global_Mask_Label": int,
        }
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_MaxIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_MeanIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_MinIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_StdIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_MedianIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_BorderMeanIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_BorderMinIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_BorderMaxIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_BorderStdIntensity"] = float
        for channel_name in channel_names:
            meta[f"Intensity_{mask_name}_{channel_name}_BorderMedianIntensity"] = float
        return meta
