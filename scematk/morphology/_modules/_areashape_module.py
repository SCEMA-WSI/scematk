from typing import List, Optional

import pandas as pd
from dask.delayed import delayed
from numpy import ndarray
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border

from ._abc_module import _ABCModule


class AreaShapeModule(_ABCModule):
    def __init__(self, areashape_metric: str = "micron", **kwargs):
        """Constructor for area shape measurment module

        Args:
            areashape_metric (str, optional): The metric to use for distance measurements. Can be 'pixel' or 'micron'. Defaults to 'micron'.
        """
        super().__init__("AreaShape")
        assert areashape_metric in [
            "micron",
            "pixel",
        ], 'Metric should be either "micron" or "pixel"'
        self.metric = areashape_metric

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
        """Measure the morphological properties of cells in a tile

        Args:
            image (ndarray): The image tile to measure.
            mask (ndarray): The mask tile to measure.
            y_seed (int): The y-coordinate of the top left corner of the tile.
            x_seed (int): the x-coordinate of the top left corner of the tile.
            channel_names (List[str]): The names of the channels in the image.
            mask_name (str): The name of the mask.
            image_mpp (Optional[float], optional): The microns per pixel of the image. Defaults to None.

        Returns:
            pd.DataFrame: The morphological properties of the cells in the tile.
        """
        image_mpp2 = image_mpp**2 if image_mpp is not None else None
        mask = clear_border(mask)
        data = regionprops_table(
            mask,
            properties=(
                "label",
                "area",
                "area_convex",
                "area_filled",
                "axis_major_length",
                "axis_minor_length",
                "eccentricity",
                "equivalent_diameter_area",
                "euler_number",
                "feret_diameter_max",
                "inertia_tensor",
                "inertia_tensor_eigvals",
                "moments_hu",
                "perimeter",
                "perimeter_crofton",
                "solidity",
            ),
        )
        data = pd.DataFrame(data)
        data.rename(
            columns={
                "label": "Meta_Global_Mask_Label",
                "area": f"{self.name}_{mask_name}_Mask_Area",
                "area_convex": f"{self.name}_{mask_name}_Mask_AreaConvex",
                "area_filled": f"{self.name}_{mask_name}_Mask_AreaFilled",
                "axis_major_length": f"{self.name}_{mask_name}_Mask_AxisMajorLength",
                "axis_minor_length": f"{self.name}_{mask_name}_Mask_AxisMinorLength",
                "eccentricity": f"{self.name}_{mask_name}_Mask_Eccentricity",
                "equivalent_diameter_area": f"{self.name}_{mask_name}_Mask_EquivalentDiameterArea",
                "euler_number": f"{self.name}_{mask_name}_Mask_EulerNumber",
                "feret_diameter_max": f"{self.name}_{mask_name}_Mask_FeretDiameterMax",
                "inertia_tensor-0-0": f"{self.name}_{mask_name}_Mask_InertiaTensor-0-0",
                "inertia_tensor-0-1": f"{self.name}_{mask_name}_Mask_InertiaTensor-0-1",
                "inertia_tensor-1-0": f"{self.name}_{mask_name}_Mask_InertiaTensor-1-0",
                "inertia_tensor-1-1": f"{self.name}_{mask_name}_Mask_InertiaTensor-1-1",
                "inertia_tensor_eigvals-0": f"{self.name}_{mask_name}_Mask_InertiaTensorEigvals-0",
                "inertia_tensor_eigvals-1": f"{self.name}_{mask_name}_Mask_InertiaTensorEigvals-1",
                "moments_hu-0": f"{self.name}_{mask_name}_Mask_MomentsHu-0",
                "moments_hu-1": f"{self.name}_{mask_name}_Mask_MomentsHu-1",
                "moments_hu-2": f"{self.name}_{mask_name}_Mask_MomentsHu-2",
                "moments_hu-3": f"{self.name}_{mask_name}_Mask_MomentsHu-3",
                "moments_hu-4": f"{self.name}_{mask_name}_Mask_MomentsHu-4",
                "moments_hu-5": f"{self.name}_{mask_name}_Mask_MomentsHu-5",
                "moments_hu-6": f"{self.name}_{mask_name}_Mask_MomentsHu-6",
                "perimeter": f"{self.name}_{mask_name}_Mask_Perimeter",
                "perimeter_crofton": f"{self.name}_{mask_name}_Mask_PerimeterCrofton",
                "solidity": f"{self.name}_{mask_name}_Mask_Solidity",
            },
            inplace=True,
        )
        if self.metric == "micron" and image_mpp is not None:
            data[f"{self.name}_{mask_name}_Mask_Area"] = (
                data[f"{self.name}_{mask_name}_Mask_Area"] * image_mpp2
            )
            data[f"{self.name}_{mask_name}_Mask_AreaConvex"] = (
                data[f"{self.name}_{mask_name}_Mask_AreaConvex"] * image_mpp2
            )
            data[f"{self.name}_{mask_name}_Mask_AreaFilled"] = (
                data[f"{self.name}_{mask_name}_Mask_AreaFilled"] * image_mpp2
            )
            data[f"{self.name}_{mask_name}_Mask_AxisMajorLength"] = (
                data[f"{self.name}_{mask_name}_Mask_AxisMajorLength"] * image_mpp
            )
            data[f"{self.name}_{mask_name}_Mask_AxisMinorLength"] = (
                data[f"{self.name}_{mask_name}_Mask_AxisMinorLength"] * image_mpp
            )
            data[f"{self.name}_{mask_name}_Mask_EquivalentDiameterArea"] = (
                data[f"{self.name}_{mask_name}_Mask_EquivalentDiameterArea"] * image_mpp
            )
            data[f"{self.name}_{mask_name}_Mask_FeretDiameterMax"] = (
                data[f"{self.name}_{mask_name}_Mask_FeretDiameterMax"] * image_mpp
            )
            data[f"{self.name}_{mask_name}_Mask_Perimeter"] = (
                data[f"{self.name}_{mask_name}_Mask_Perimeter"] * image_mpp
            )
            data[f"{self.name}_{mask_name}_Mask_PerimeterCrofton"] = (
                data[f"{self.name}_{mask_name}_Mask_PerimeterCrofton"] * image_mpp
            )
        return data

    def _get_meta(self, channel_names: List[str], mask_name: str) -> dict:
        """Get the column metadata

        Args:
            channel_names (List[str]): the names of the channels in the image.
            mask_name (str): The name of the mask.

        Returns:
            dict: A dictionary of the column metadata.
        """
        meta = {
            "Meta_Global_Mask_Label": int,
            f"{self.name}_{mask_name}_Mask_Area": float,
            f"{self.name}_{mask_name}_Mask_AreaConvex": float,
            f"{self.name}_{mask_name}_Mask_AreaFilled": float,
            f"{self.name}_{mask_name}_Mask_AxisMajorLength": float,
            f"{self.name}_{mask_name}_Mask_AxisMinorLength": float,
            f"{self.name}_{mask_name}_Mask_Eccentricity": float,
            f"{self.name}_{mask_name}_Mask_EquivalentDiameterArea": float,
            f"{self.name}_{mask_name}_Mask_EulerNumber": int,
            f"{self.name}_{mask_name}_Mask_FeretDiameterMax": float,
            f"{self.name}_{mask_name}_Mask_InertiaTensor-0-0": float,
            f"{self.name}_{mask_name}_Mask_InertiaTensor-0-1": float,
            f"{self.name}_{mask_name}_Mask_InertiaTensor-1-0": float,
            f"{self.name}_{mask_name}_Mask_InertiaTensor-1-1": float,
            f"{self.name}_{mask_name}_Mask_InertiaTensorEigvals-0": float,
            f"{self.name}_{mask_name}_Mask_InertiaTensorEigvals-1": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-0": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-1": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-2": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-3": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-4": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-5": float,
            f"{self.name}_{mask_name}_Mask_MomentsHu-6": float,
            f"{self.name}_{mask_name}_Mask_Perimeter": float,
            f"{self.name}_{mask_name}_Mask_PerimeterCrofton": float,
            f"{self.name}_{mask_name}_Mask_Solidity": float,
        }
        return meta
