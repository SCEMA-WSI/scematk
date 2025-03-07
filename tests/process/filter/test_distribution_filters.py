import numpy as np
from scipy.ndimage import (
    maximum_filter,
    median_filter,
    minimum_filter,
    percentile_filter,
)
from skimage.morphology import disk

from scematk.io import read_zarr_ubimg
from scematk.process.filter import (
    MaximumFilter,
    MedianFilter,
    MinimumFilter,
    PercentileFilter,
)


class TestMaximumFilter:

    def test_maximum_filter_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        filtered_image = MaximumFilter(3).run(image).image.compute()
        footprint = disk(image.pixel_from_micron(3))
        footprint = np.expand_dims(footprint, axis=-1)
        ref_image = maximum_filter(image.image.compute(), footprint=footprint).astype(np.uint8)
        assert np.array_equal(filtered_image, ref_image)
        assert filtered_image.shape == ref_image.shape

    def test_maximum_filter_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = MaximumFilter(3).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names
        

class TestMedianFilter:

    def test_median_filter_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        filtered_image = MedianFilter(3).run(image).image.compute()
        footprint = disk(image.pixel_from_micron(3))
        footprint = np.expand_dims(footprint, axis=-1)
        ref_image = median_filter(image.image.compute(), footprint=footprint).astype(np.uint8)
        assert np.array_equal(filtered_image, ref_image)
        assert filtered_image.shape == ref_image.shape

    def test_median_filter_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = MedianFilter(3).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names


class TestMinimumFilter:

    def test_minimum_filter_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        filtered_image = MinimumFilter(3).run(image).image.compute()
        footprint = disk(image.pixel_from_micron(3))
        footprint = np.expand_dims(footprint, axis=-1)
        ref_image = minimum_filter(image.image.compute(), footprint=footprint).astype(np.uint8)
        assert np.array_equal(filtered_image, ref_image)
        assert filtered_image.shape == ref_image.shape

    def test_minimum_filter_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = MinimumFilter(3).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names

class TestPercentileFilter:

    def test_percentile_filter_image(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        filtered_image = PercentileFilter(3, 53).run(image).image.compute()
        footprint = disk(image.pixel_from_micron(3))
        footprint = np.expand_dims(footprint, axis=-1)
        ref_image = percentile_filter(image.image.compute(), 53, footprint=footprint).astype(np.uint8)
        assert np.array_equal(filtered_image, ref_image)
        assert filtered_image.shape == ref_image.shape

    def test_percentile_filter_meta(self):
        image = read_zarr_ubimg(
            "./tests/_testdata/raw_img/image.zarr", "./tests/_testdata/raw_img/meta.json"
        )
        contrast_image = PercentileFilter(3, 53).run(image)
        assert image.info == contrast_image.info
        assert image.channel_names == contrast_image.channel_names