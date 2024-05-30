import json
import os
from itertools import product

import zarr
from openslide import OpenSlide
from tqdm import tqdm


def tiff_to_zarr(
    tiff_path: str, zarr_path: str, meta_path: str, tile_size: int = 4096, chunk_size: int = 4096
) -> None:
    """Convert a TIFF image to a Zarr array and JSON metadata file.

    Args:
        tiff_path (str): Path to the TIFF image.
        zarr_path (str): Path to save the Zarr array.
        meta_path (str): Path to save the JSON metadata file.
        tile_size (int, optional): Size of the tiles to read from the TIFF image. Defaults to 4096.
        chunk_size (int, optional): Size of the chunks to write to the Zarr array. Defaults to 4096.
    """
    assert isinstance(tiff_path, str), "tiff_path must be a string"
    assert os.path.exists(tiff_path), "tiff_path does not exist"
    assert isinstance(zarr_path, str), "zarr_path must be a string"
    assert not os.path.exists(zarr_path), "zarr_path already exists"
    assert isinstance(meta_path, str), "meta_path must be a string"
    assert not os.path.exists(meta_path), "meta_path already exists"
    assert isinstance(tile_size, int), "tile_size must be an integer"
    assert tile_size > 0, "tile_size must be positive"
    assert isinstance(chunk_size, int), "chunk_size must be an integer"
    assert chunk_size > 0, "chunk_size must be positive"
    with OpenSlide(tiff_path) as slide:
        slide_dims = slide.dimensions
        out_shape = [slide_dims[1], slide_dims[0], 3]
        out_img = zarr.zeros(
            out_shape,
            dtype="uint8",
            chunks=(chunk_size, chunk_size, 3),
            store=zarr.DirectoryStore(zarr_path),
        )
        y_mins = range(0, out_shape[0], tile_size)
        x_mins = range(0, out_shape[1], tile_size)
        num_iter = len(y_mins) * len(x_mins)
        for y_min, x_min in tqdm(product(y_mins, x_mins), total=num_iter):
            y_max = min(y_min + tile_size, out_shape[0])
            x_max = min(x_min + tile_size, out_shape[1])
            tile = slide.read_region((x_min, y_min), 0, (x_max - x_min, y_max - y_min)).convert(
                "RGB"
            )
            out_img[y_min:y_max, x_min:x_max] = tile
        slide_metadata = {
            "name": os.path.basename(tiff_path),
            "format": str(slide.detect_format(tiff_path)),
        }
        if (
            hasattr(slide, "properties")
            and "openslide.mpp-x" in slide.properties
            and "openslide.mpp-y" in slide.properties
        ):
            slide_metadata["mpp-x"] = str(float(slide.properties["openslide.mpp-x"]))
            slide_metadata["mpp-y"] = str(float(slide.properties["openslide.mpp-y"]))
            if slide_metadata["mpp-x"] == slide_metadata["mpp-y"]:
                slide_metadata["mpp"] = slide_metadata["mpp-x"]
        with open(meta_path, "w") as f:
            json.dump(slide_metadata, f)
