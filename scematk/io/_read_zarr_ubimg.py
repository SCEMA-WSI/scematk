import json
import os
from typing import List

import dask.array as da

from ..image._ubyte_image import UByteImage


def read_zarr_ubimg(
    zarr_path: str, meta_path: str, channel_names: List[str] | None = None
) -> UByteImage:
    """Read a Zarr array and JSON metadata file into a UByteImage.

    Args:
        zarr_path (str): Path to the Zarr array.
        meta_path (str): Path to the JSON metadata file.
        channel_names (List[str], optional): Names of the channels. Defaults to None.

    Raises:
        NotImplementedError: Images of type other than uint8 are not supported yet.

    Returns:
        UByteImage: UByteImage object.
    """
    assert isinstance(zarr_path, str)
    assert zarr_path.endswith(".zarr")
    assert os.path.exists(zarr_path)
    assert isinstance(meta_path, str)
    assert meta_path.endswith(".json")
    assert os.path.exists(meta_path)
    img = da.from_zarr(zarr_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    if channel_names is None:
        channel_names = ["Red", "Green", "Blue"]
    if str(img.dtype) == "uint8":
        return UByteImage(img, meta, channel_names)
    else:
        raise NotImplementedError(f"Images of type {str(img.dtype)} are not supported yet.")
