import os
from typing import Tuple

import numpy as np
import zarr
from csbdeep.data import Normalizer, normalize_mi_ma
from stardist.models import StarDist2D


def segment_stardist(
    in_path: str,
    out_path: str,
    model: str = "2D_versatile_he",
    axes: str = "YXC",
    block_size: int = 4096,
    min_overlap: int = 128,
    context: int = 128,
    n_tiles: Tuple[int, int, int] = (4, 4, 1),
    prob_thresh: float = 0.2,
    nms_thresh: float = 0.3,
    skip_warning: bool = False,
) -> None:
    """
    Segment an image using the StarDist model.

    Parameters:
        in_path (str): Path to the input zarr file.
        out_path (str): Path to the output zarr file.
        model (str): Model to use. Default is '2D_versatile_he'.
        axes (str): Axes of the input image. Default is 'YXC'.
        block_size (int): Size of the blocks to process. Default is 4096.
        min_overlap (int): Minimum overlap between blocks. Default is 128.
        context (int): Context for the prediction. Default is 128.
        n_tiles (tuple): Number of tiles to use. Default is (4,4,1).
        prob_thresh (float): Probability threshold. Default is 0.2.
        nms_thresh (float): NMS threshold. Default is 0.3.
        skip_warning (bool): Skip the warning message. Default is False.

    Returns:
        None

    Raises:
        AssertionError: If any of the input arguments are invalid.
        ValueError: If the user chooses to exit.
    """
    assert isinstance(in_path, str), "Input path must be a string"
    assert os.path.exists(in_path), "Input path does not exist"
    assert isinstance(out_path, str), "Output path must be a string"
    assert not os.path.exists(out_path), "Output path already exists"
    assert isinstance(model, str), "Model must be a string"
    assert model in ["2D_versatile_he", "2D_versatile_fluo", "2D_paper_dsb2018", "2D_demo"]
    assert isinstance(axes, str), "Axes must be a string"
    assert axes in ["YXC", "CYX"]
    assert isinstance(block_size, int), "Block size must be an integer"
    assert block_size > 0, "Block size must be greater than 0"
    assert isinstance(min_overlap, int), "Minimum overlap must be an integer"
    assert min_overlap > 0, "Minimum overlap must be greater than 0"
    assert isinstance(context, int), "Context must be an integer"
    assert context > 0, "Context must be greater than 0"
    assert isinstance(n_tiles, tuple), "Number of tiles must be a tuple"
    assert len(n_tiles) == 3, "Number of tiles must have 3 elements"
    assert all(isinstance(i, int) for i in n_tiles), "Number of tiles must be integers"
    assert all(i > 0 for i in n_tiles), "Number of tiles must be greater than 0"
    assert isinstance(prob_thresh, float), "Probability threshold must be a float"
    assert 0 <= prob_thresh <= 1, "Probability threshold must be between 0 and 1"
    assert isinstance(nms_thresh, float), "NMS threshold must be a float"
    assert 0 <= nms_thresh <= 1, "NMS threshold must be between 0 and 1"
    assert isinstance(skip_warning, bool), "Skip warning must be a boolean"
    if not skip_warning:
        response = input(
            """
            Warning: This function is not implemented using Dask for parallel processing. This may result in slow processing times for large images.
            This is an implementation of the pretrained StarDist model as set out in their examples for large images.
            Would you like to continue? (y/[n])
            """
        )
        response = response.strip().lower()
        if response != "y":
            raise ValueError("User chose to exit")

    class MyNormalizer(Normalizer):
        def __init__(self, mi, ma):
            self.mi, self.ma = mi, ma

        def before(self, x, axes):
            return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)

        def after(*args, **kwargs):
            assert False

        @property
        def do_after(self):
            return False

    mi, ma = 0, 255
    normalizer = MyNormalizer(mi, ma)
    image = zarr.open(in_path)
    out_zarr = zarr.zeros(shape=image.shape[:2], chunks=image.chunks[:2], dtype=np.int32)
    sd_model = StarDist2D.from_pretrained(model)
    _, _ = sd_model.predict_instances_big(
        image,
        axes=axes,
        block_size=block_size,
        min_overlap=min_overlap,
        context=context,
        normalizer=normalizer,
        n_tiles=n_tiles,
        labels_out=out_zarr,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    zarr.save(out_path, out_zarr)
