from .._process import Process
from dask.array import Array
from dask_image.ndfilters import gaussian_filter

class GaussianFilter(Process):
    def __init__(self, sigma, order=0, mode='reflect', cval=0.0, truncate=4.0):
        super().__init__("Apply Gaussian filter")
        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.cval = cval
        self.truncate = truncate

    def process(self, image: Array) -> Array:
        assert isinstance(image, Array), f"Expected image to be of type Array, got {type(image)}"
        sigma = self.sigma
        order = self.order
        mode = self.mode
        cval = self.cval
        truncate = self.truncate
        return gaussian_filter(image, sigma=sigma, order=order, mode=mode, cval=cval, truncate=truncate)