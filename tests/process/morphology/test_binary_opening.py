import numpy as np

from scematk.io import read_zarr_bin_mask
from scematk.process.morphology import BinaryOpening
from scipy.ndimage import binary_opening
from skimage.morphology import disk

class TestBinaryOpening:
    
    def test_binary_opening_image(self):
        image = read_zarr_bin_mask(
            "./tests/_testdata/tis_mask/image.zarr", "./tests/_testdata/tis_mask/meta.json", mask_name="TissueMask"
        )
        contrast_image = BinaryOpening(3).run(image).image.compute()
        ref_image = image.image.compute()
        footprint = disk(image.pixel_from_micron(3))
        ref_image = binary_opening(ref_image, structure=footprint)
        assert np.array_equal(contrast_image, ref_image)

    def test_binary_opening_meta(self):
        image = read_zarr_bin_mask(
            "./tests/_testdata/tis_mask/image.zarr", "./tests/_testdata/tis_mask/meta.json", mask_name="TissueMask"
        )
        contrast_image = BinaryOpening(3).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names