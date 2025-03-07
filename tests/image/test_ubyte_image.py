import dask.array as da
import numpy as np

from scematk.io import read_zarr_ubimg


class TestUByteImage:

    def test_dask_array(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        assert isinstance(image.image, da.Array)
        assert image.image.ndim == 3
        assert image.image.shape == (2967, 2220, 3)
        assert image.image.dtype == np.uint8

    def test_image_metadata(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        assert isinstance(image.info, dict)
        assert image.info["name"] == "small_tiff.tiff"
        assert image.info["format"] == "aperio"
        assert image.info["mpp-x"] == "0.499"
        assert image.info["mpp-y"] == "0.499"
        assert image.info["mpp"] == "0.499"
        assert image.channel_names == ["Red", "Green", "Blue"]
