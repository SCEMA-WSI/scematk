from ..image._binary_mask import BinaryMask
import dask.array as da
import json
import os
from typing import List

def read_zarr_bin_mask(zarr_path: str, meta_path: str, mask_name: List[str] | str | None = None) -> BinaryMask:
    """Read a Zarr array and JSON metadata file into a BinaryMask.

    Args:
        zarr_path (str): Path to the Zarr array.
        meta_path (str): Path to the JSON metadata file.
        mask_name (List[str], str, optional): Name of the mask. Defaults to None. 

    Returns:
        BinaryMask: BinaryMask object.
    """
    assert isinstance(zarr_path, str)
    assert zarr_path.endswith('.zarr')
    assert os.path.exists(zarr_path)
    assert isinstance(meta_path, str)
    assert meta_path.endswith('.json')
    assert os.path.exists(meta_path)
    assert isinstance(mask_name, (list, str)) or mask_name is None
    if mask_name is None:
        mask_name = ['Mask']
    if not isinstance(mask_name, list):
        mask_name = [mask_name]
    mask = da.from_zarr(zarr_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return BinaryMask(mask, meta, channel_names=mask_name)