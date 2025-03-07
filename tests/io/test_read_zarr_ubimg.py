import pytest

from scematk.image._ubyte_image import UByteImage
from scematk.io import read_zarr_ubimg


class TestReadZarrUBImg:

    def test_reading_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        assert isinstance(image, UByteImage)
