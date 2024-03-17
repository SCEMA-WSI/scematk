import dask.array as da
from scematk.plot import show_thumb

def test_int_show_thumb():
    img = da.random.randint(0, 255, size=(100, 100, 3), chunks=(50, 50, 3))
    show_thumb(img)
    assert True