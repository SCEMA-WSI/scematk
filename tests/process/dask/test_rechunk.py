import numpy as np

from scematk.io import read_zarr_ubimg
from scematk.process.dask import Rechunk


class TestRechunk:

    def test_gamma_contrast_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        rechunked_image = Rechunk((200, 200, 3)).run(image).image
        rechunked_image_c = rechunked_image.compute()
        ref_image = image.image.compute()
        assert np.array_equal(rechunked_image_c, ref_image)

    def test_gamma_contrast_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = Rechunk((200, 200, 3)).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names
