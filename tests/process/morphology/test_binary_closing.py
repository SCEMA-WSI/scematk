import numpy as np

from scematk.io import read_zarr_bin_mask
from scematk.process.morphology import BinaryClosing
from scipy.ndimage import binary_closing
from skimage.morphology import disk

class TestBinaryClosing:
    
    def test_binary_closing_image(self):
        image = read_zarr_bin_mask(
            "./tests/_testdata/tis_mask/image.zarr", "./tests/_testdata/tis_mask/meta.json", mask_name="TissueMask"
        )
        contrast_image = BinaryClosing(3).run(image).image.compute()
        ref_image = image.image.compute()
        footprint = disk(image.pixel_from_micron(3))
        ref_image = binary_closing(ref_image, structure=footprint)
        assert np.array_equal(contrast_image, ref_image)

    def test_binary_closing_meta(self):
        image = read_zarr_bin_mask(
            "./tests/_testdata/tis_mask/image.zarr", "./tests/_testdata/tis_mask/meta.json", mask_name="TissueMask"
        )
        contrast_image = BinaryClosing(3).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names