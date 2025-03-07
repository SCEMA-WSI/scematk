import numpy as np
from skimage import img_as_ubyte
from skimage.color import rgb2gray

from scematk.io import read_zarr_ubimg
from scematk.process.colour import RGBToGrey


class TestRGBToGrey:

    def test_rgb_to_grey_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        grey_image = RGBToGrey().run(image).image.compute()
        ref_image = image.image.compute()
        ref_image = img_as_ubyte(rgb2gray(ref_image))
        ref_image = np.expand_dims(ref_image, axis=-1)
        assert np.array_equal(grey_image, ref_image)

    def test_rgb_to_grey_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        grey_image = RGBToGrey().run(image)
        assert image.info == grey_image.info
        assert grey_image.channel_names == ["Grey"]
