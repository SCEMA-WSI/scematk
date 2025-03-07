import numpy as np

from scematk.io import read_zarr_ubimg
from scematk.process.filter import GaussianBlur
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk

class TestGaussianBlur:

    def test_gaussian_blur_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        filtered_image = GaussianBlur(3).run(image).image.compute()
        footprint = disk(image.pixel_from_micron(3))
        footprint = np.expand_dims(footprint, axis=-1)
        sigma = image.pixel_from_micron(3)
        ref_image = gaussian_filter(image.image.compute().astype("float32"), sigma = (sigma, sigma, 0), mode="mirror", truncate=3.0).astype(np.uint8)
        assert np.array_equal(filtered_image, ref_image)

    def test_gaussian_blur_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = GaussianBlur(3).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names