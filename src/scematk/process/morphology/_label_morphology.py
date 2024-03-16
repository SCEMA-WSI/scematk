from .._process import Process
from dask.array import Array
import dask.array as da
from skimage.segmentation import expand_labels

class LabelDilation(Process):
    def __init__(self, distance=1):
        super().__init__("Perform label dilation")
        self.distance = distance

    def process(self, label_image: Array) -> Array:
        assert isinstance(label_image, Array), f"Expected label_image to be of type Array, got {type(label_image)}"
        assert label_image.ndim == 2, f"Expected label_image to be 2D, got {label_image.ndim}D"
        assert label_image.dtype in (int, 'int32', 'int64'), f"Expected label_image to be of type int, got {label_image.dtype}"
        distance = self.distance
        overlap = 2 * distance + 1
        label_image = da.map_overlap(lambda x: expand_labels(x, distance=distance), label_image, depth=overlap, boundary='none', dtype=label_image.dtype)
        return label_image