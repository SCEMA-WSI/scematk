from .._process import Process
import dask.array as da
from dask.array import Array

class ABContrast2D(Process):
    def __init__(self, a: float = 1, b: float = 0):
        assert isinstance(a, float), "a must be a float"
        assert isinstance(b, float), "b must be a float"
        self.a = a
        self.b = b

    def process(self, image: Array) -> Array:
        image = image * self.a + self.b
        image = da.clip(image, 0, 255)
        image = image.astype('uint8')
        return image

class GammaContrast2D(Process):
    def __init__(self, gamma: float = 1):
        assert isinstance(gamma, float), "gamma must be a float"
        self.gamma = gamma

    def process(self, image: Array) -> Array:
        assert isinstance(image, da.Array), "image must be a dask array"
        assert image.ndim == 2, "image must be a 2D array"
        assert image.dtype == 'uint8', "image must be a uint8 array"
        image = image / 255
        image = image ** self.gamma
        image = image * 255
        image = image.astype('uint8')
        return image