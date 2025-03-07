import numpy as np

from scematk.io import read_zarr_ubimg
from scematk.process.contrast import GammaContrast


class TestGammaContrast:

    def test_gamma_contrast_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = GammaContrast(2).run(image).image.compute()
        ref_image = image.image.compute()
        ref_image = ref_image / 255
        ref_image = ref_image**2
        ref_image = ref_image * 255
        ref_image = np.floor(ref_image).astype(np.uint8)
        assert np.array_equal(contrast_image, ref_image)

    def test_gamma_contrast_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = GammaContrast(2).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names
