from typing import Optional

import dask.array as da
import numpy as np
from skimage import img_as_ubyte
from skimage.color import lab2rgb, rgb2lab

from ..image._image import Image
from ..image._ubyte_image import UByteImage
from ..process._process import Processor
from ._normaliser import Normaliser


class ReinhardNormaliser(Normaliser):
    def __init__(
        self,
        preprocessor: Optional[Processor] = None,
        postprocessor: Optional[Processor] = None,
        luminosity_thresh: float = 70,
    ) -> None:
        super().__init__("Reinhard Normaliser", preprocessor, postprocessor)
        assert isinstance(luminosity_thresh, (int, float)), "luminosity_thresh must be a number"
        assert 0 < luminosity_thresh < 100, "luminosity_thresh must be between 0 and 100"
        self.luminosity_thresh = luminosity_thresh

    def fit(self, image: Image) -> None:
        image = self.preprocessor.run(image)
        img = image.image
        lab_img = da.map_blocks(rgb2lab, img, dtype=float)
        mask = lab_img[:, :, 0] > self.luminosity_thresh
        mask = da.stack([mask, mask, mask], axis=2)
        masked_lab_img = da.ma.masked_array(lab_img, mask)
        means = da.mean(masked_lab_img, axis=(0, 1))
        stds = da.std(masked_lab_img, axis=(0, 1))
        outs = da.stack([means, stds], axis=1).compute().data
        self.means = outs[:, 0]
        self.stds = outs[:, 1]
        self.fitted = True

    def run(self, image: Image) -> Image:
        assert self.fitted, "Fit the normaliser before running it"
        image = self.preprocessor.run(image)
        img = image.image
        lab_img = da.map_blocks(rgb2lab, img, dtype=float)
        mask = lab_img[:, :, 0] > self.luminosity_thresh
        mask = da.stack([mask, mask, mask], axis=2)
        masked_lab_img = da.ma.masked_array(lab_img, mask)
        means = da.mean(masked_lab_img, axis=(0, 1))
        stds = da.std(masked_lab_img, axis=(0, 1))
        outs = da.ma.getdata(da.stack([means, stds], axis=1))
        old_means = outs[:, 0]
        old_stds = outs[:, 1]
        new_means = self.means
        new_stds = self.stds
        normalised_lab_img = (lab_img - old_means) * new_stds / old_stds + new_means
        normalised_float_img = da.map_blocks(lab2rgb, normalised_lab_img, dtype=float)
        normalised_float_img = da.clip(normalised_float_img, 0, 1)
        normalised_img = da.map_blocks(img_as_ubyte, normalised_float_img, dtype=np.uint8)
        norm_img = UByteImage(normalised_img, image.info, image.channel_names)
        return self.postprocessor.run(norm_img)

    def fit_and_run(self, image: Image) -> Image:
        assert False, "You can't normalise an image with itself"

    def _default_preprocessor(self) -> Processor:
        return Processor()

    def _default_postprocessor(self) -> Processor:
        return Processor()
