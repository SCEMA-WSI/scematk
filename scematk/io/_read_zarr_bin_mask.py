from ..image._binary_mask import BinaryMask
import dask.array as da
import json
import os

def read_zarr_bin_mask(zarr_path: str, meta_path: str, mask_name: str = None) -> BinaryMask:
    assert isinstance(zarr_path, str)
    assert zarr_path.endswith('.zarr')
    assert os.path.exists(zarr_path)
    assert isinstance(meta_path, str)
    assert meta_path.endswith('.json')
    assert os.path.exists(meta_path)
    if mask_name is None:
        mask_name = 'Mask'
    mask = da.from_zarr(zarr_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return BinaryMask(mask, meta, channel_names='Mask')