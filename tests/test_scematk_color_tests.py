import dask.array as da
from scematk.colour import rgb_to_grey
from skimage.color import rgb2gray

def test_rgb_to_grey():
    rimg = da.random.randint(0, 255, size=(500, 500, 3), chunks=(50, 50, 3))
    assert rgb_to_grey(rimg).compute() == rgb2gray(rimg.compute())

def test_int_rgb_to_grey():
    rimg = da.random.randint(0, 255, size=(500, 500, 3), chunks=(50, 50, 3), dtype='int')
    assert isinstance(rgb_to_grey(rimg), da.Array)
