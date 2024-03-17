import dask.array as da
import numpy as np
from scematk.process.colour import RGBToGrey
from skimage import img_as_ubyte
from skimage.color import rgb2gray

def test_RGBToGrey():
    img = np.random.randint(0, 255, size=(100, 100, 3), dtype="uint8")
    ref = img_as_ubyte(rgb2gray(img))
    img = da.from_array(img, chunks=(50, 50, 3))
    test = RGBToGrey().process(img).compute()
    equality = ref == test
    assert equality.all()