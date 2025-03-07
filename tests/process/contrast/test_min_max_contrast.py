import numpy as np

from scematk.io import read_zarr_ubimg
from scematk.process.contrast import MinMaxContrast


class TestMinMaxContrast:

    def test_min_max_contrast_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = MinMaxContrast().run(image).image.compute()
        ref_image = image.image.compute()
        ref_image = ref_image - ref_image.min()
        ref_image = ref_image / ref_image.max()
        ref_image = ref_image * 255
        ref_image = np.floor(ref_image).astype(np.uint8)
        assert np.array_equal(contrast_image, ref_image)

    def test_min_max_contrast_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = MinMaxContrast().run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names
