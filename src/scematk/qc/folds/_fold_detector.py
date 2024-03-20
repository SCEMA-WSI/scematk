from .._qc_step import QCStep
from ...process._processor import Processor
import dask.array as da
from dask.array import Array
from dask_image.ndmorph import binary_dilation
import numpy as np
from skimage.color import rgb2hsv

class FoldDetector(QCStep):
    def __init__(self, name: str, preprocessor: Processor = None, postprocessor: Processor = None) -> None:
        super().__init__(name, preprocessor, postprocessor)
        self.fitted = True

    def fit(self, image: Array) -> None:
        pass

    def run(self, image: Array) -> Array:
        image = da.map_blocks(rgb2hsv, image, dtype=float)
        delta = image[:, :, 1] - image[:, :, 2]
        delta_mask = da.greater(delta, 0)
        structure = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ])
        delta_mask_dilated = binary_dilation(delta_mask, structure=structure)
        delta_mask_lower = da.greater(delta, -0.3)
        delta_mask_dilated = da.logical_and(delta_mask_dilated, delta_mask_lower)
        return delta_mask_dilated

    def fit_and_run(self, image: Array) -> Array:
        return self.run(image)

    def _default_preprocessor(self) -> Processor:
        return Processor()

    def _default_postprocessor(self) -> Processor:
        return Processor()
