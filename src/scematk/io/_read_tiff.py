from ..wsi._wsi import WSI
import dask.array as da
from openslide import OpenSlide
import os
from tifffile import imread
import zarr

def read_tiff(path: str, level: int = 0) -> WSI:
    assert isinstance(path, str), 'path must be a string'
    assert os.path.exists(path), f'path does not exist: {path}'
    assert isinstance(level, int), 'level must be an integer'
    try:
        slide = OpenSlide(path)
        metadata = {
            "format": str(slide.detect_format(path)),
            "level_count": int(slide.level_count),
        }

        if 'openslide.mpp-x' in slide.properties:
            metadata["mpp-x"] = float(slide.properties['openslide.mpp-x'])
        if 'openslide.mpp-y' in slide.properties:
            metadata["mpp-y"] = float(slide.properties['openslide.mpp-y'])
        if metadata.get("mpp-x") and metadata.get("mpp-y"):
            if metadata["mpp-x"] == metadata["mpp-y"]:
                metadata["mpp"] = metadata["mpp-x"]
        slide.close()

        img = imread(path, aszarr=True)
        img = zarr.open(img)
        img = da.from_zarr(img[level])
        return WSI(img, metadata)

    except Exception as e:
        raise e
        with OpenSlide(path) as slide:
            metadata = {
                "format": slide.detect_format(path),
                "level_count": slide.level_count,
                "level_dimensions": [slide.level_dimensions[i] for i in range(slide.level_count)],
                "level_downsamples": [slide.level_downsamples[i] for i in range(slide.level_count)],
                "properties": slide.properties,
            }

            if 'openslide.mpp-x' in slide.properties:
                metadata["mpp-x"] = float(slide.properties['openslide.mpp-x'])
            if 'openslide.mpp-y' in slide.properties:
                metadata["mpp-y"] = float(slide.properties['openslide.mpp-y'])
            if metadata.get("mpp-x") and metadata.get("mpp-y"):
                if metadata["mpp-x"] == metadata["mpp-y"]:
                    metadata["mpp"] = metadata["mpp-x"]

            img = imread(path, aszarr=True)
            img = zarr.open(img)
            img = da.from_zarr(img[level])
            return WSI(img, metadata)

    except Exception as e:
        raise e