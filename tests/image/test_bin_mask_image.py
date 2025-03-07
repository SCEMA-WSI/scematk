import dask.array as da
import numpy as np

from scematk.io import read_zarr_bin_mask


class TestBinMaskImage:

    def test_dask_array(self):
        image = read_zarr_bin_mask(
            "./tests/_testdata/tis_mask/image.zarr",
            "./tests/_testdata/tis_mask/meta.json",
            mask_name="Tissue",
        )
        assert isinstance(image.image, da.Array)
        assert image.image.ndim == 2
        assert image.image.shape == (2967, 2220)
        assert image.image.dtype == bool

    def test_image_metadata(self):
        image = read_zarr_bin_mask(
            "./tests/_testdata/tis_mask/image.zarr",
            "./tests/_testdata/tis_mask/meta.json",
            mask_name="Tissue",
        )
        assert isinstance(image.info, dict)
        assert image.info["name"] == "small_tiff.tiff"
        assert image.info["format"] == "aperio"
        assert image.info["mpp-x"] == "0.499"
        assert image.info["mpp-y"] == "0.499"
        assert image.info["mpp"] == "0.499"
        assert image.channel_names == ["Tissue"]
