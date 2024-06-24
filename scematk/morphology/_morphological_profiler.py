from functools import reduce
from itertools import chain
from typing import List, Optional

import dask.array as da
import dask.dataframe as dd

from ..image._label_mask import LabelMask
from ..image._ubyte_image import UByteImage
from ._modules._abc_module import _ABCModule
from ._modules._areashape_module import AreaShapeModule
from ._modules._default_module import DefaultModule
from ._modules._intensity_module import IntensityModule


class MorphologicalProfiler:
    def __init__(
        self,
        modules: str | List[str] = "all",
        block_size: int = 4096,
        overlap_size: int = 256,
        **kwargs,
    ):
        """Constructor for morphological profiler

        Args:
            modules (str | List[str], optional): A list of measurements modules to include. Defaults to 'all'.
            block_size (int, optional): The size of the blocks to use. Defaults to 4096.
            overlap_size (int, optional): The size of the overlap to use. Defaults to 256.
        """
        modules = modules if isinstance(modules, list) else [modules]
        assert all([isinstance(module, str) for module in modules]), "All modules should be strings"
        self.modules = self._get_modules(modules, **kwargs)
        assert (
            isinstance(block_size, int) and block_size > 0
        ), "Block size should be a positive integer"
        self.block_size = block_size
        assert (
            isinstance(overlap_size, int) and overlap_size > 0
        ), "Overlap size should be a positive integer"
        self.overlap_size = overlap_size

    def _get_modules(self, modules: List[str], **kwargs) -> List[_ABCModule]:
        """Get the modules requested

        Args:
            modules (List[str]): The names of the modules requested.

        Returns:
            List[_ABCModule]: A list of the requested modules.
        """
        return_modules: List[_ABCModule] = []
        if "no-centroids" not in modules:
            return_modules.append(DefaultModule(**kwargs))
        if "all" in modules or "areashape" in modules:
            return_modules.append(AreaShapeModule(**kwargs))
        if "all" in modules or "intensity" in modules:
            return_modules.append(IntensityModule(**kwargs))
        return return_modules

    def measure(
        self,
        image: UByteImage,
        masks: LabelMask | List[LabelMask],
        channel_names: Optional[str | List[str]] = None,
        mask_names: Optional[str | List[str]] = None,
    ) -> dd.DataFrame:
        """Morphologically profile the cells in a WSI.

        Args:
            image (UByteImage): The image to morphologically profile.
            masks (LabelMask | List[LabelMask]): The mask or masks to morphologically profile.
            channel_names (Optional[str  |  List[str]], optional): The names of the channels in the image to use. Defaults to None.
            mask_names (Optional[str  |  List[str]], optional): The names of the masks to use. Defaults to None.

        Returns:
            dd.DataFrame: A dask dataframe of the morphological measurements.
        """
        assert isinstance(image, UByteImage), "Image should be an instance of UByteImage"
        masks = masks if isinstance(masks, list) else [masks]
        assert isinstance(masks, list), "Mask should be an instance of LabelMask"
        assert all(
            [isinstance(mask, LabelMask) for mask in masks]
        ), "All masks should be instances of LabelMask"
        assert all(
            [mask.image.shape[:2] == image.image.shape[:2] for mask in masks]
        ), "All masks should have the same shape as the image"
        if channel_names is None:
            channel_names = image.channel_names
        channel_names = channel_names if isinstance(channel_names, list) else [channel_names]
        assert all(
            isinstance(channel_name, str) for channel_name in channel_names
        ), "All channel names should be strings"
        assert (
            len(channel_names) == image.image.shape[2]
        ), "Number of channel names should be equal to the number of channels in the image"
        if mask_names is None:
            mask_names = [mask_item.channel_names[0] for mask_item in masks]
        mask_names = mask_names if isinstance(mask_names, list) else [mask_names]
        assert all(
            isinstance(mask_name, str) for mask_name in mask_names
        ), "All mask names should be strings"
        assert len(mask_names) == len(
            masks
        ), "Number of mask names should be equal to the number of masks"
        modules = self.modules
        block_size = self.block_size
        overlap_size = self.overlap_size
        img_img = image.image.rechunk((block_size, block_size, -1))
        img_img = da.overlap.overlap(
            img_img, depth=(overlap_size, overlap_size, 0), boundary=None
        ).to_delayed()
        dfs = []
        for mask, mask_name in zip(masks, mask_names):
            mask_img = mask.image.rechunk((block_size, block_size))
            mask_img = da.overlap.overlap(
                mask_img, depth=(overlap_size, overlap_size), boundary=None
            ).to_delayed()
            for module in modules:
                intermediate_dfs = []
                for img, msk in zip(chain.from_iterable(img_img), chain.from_iterable(mask_img)):
                    y_index = img[0].key[1]
                    x_index = img[0].key[2]
                    intermediate_dfs.append(
                        module.measure(
                            img[0],
                            msk,
                            block_size,
                            overlap_size,
                            y_index,
                            x_index,
                            channel_names,
                            mask_name,
                            image.mpp,
                        )
                    )
                dfs.append(dd.concat(intermediate_dfs))
        if len(dfs) == 1:
            return dfs[0]
        elif len(dfs) == 0:
            return None
        else:
            reduced_dd = reduce(
                lambda x, y: dd.merge(x, y, on="Meta_Global_Mask_Label", how="outer"), dfs
            )
            reduced_dd = reduced_dd.drop_duplicates(subset="Meta_Global_Mask_Label")
            return reduced_dd
