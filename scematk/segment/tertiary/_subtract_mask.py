from typing import List, Optional

from ...image._label_mask import LabelMask


def subtract_mask(
    large_mask: LabelMask, small_mask: LabelMask, mask_name: Optional[List[str] | str] = None
) -> LabelMask:
    """Subtract a small mask from a large mask

    Args:
        large_mask (LabelMask): The large mask to subtract from
        small_mask (LabelMask): The small mask to subtract
        mask_name (Optional[List[str]  |  str], optional): _description_. Defaults to None.

    Returns:
        LabelMask: _description_
    """
    assert isinstance(large_mask, LabelMask), "large_mask must be a LabelMask"
    assert isinstance(small_mask, LabelMask), "small_mask must be a LabelMask"
    if mask_name is None:
        mask_name = "Mask"
    if isinstance(mask_name, str):
        mask_name = [mask_name]
    assert isinstance(mask_name, list), "mask_name must be a list of strings"
    assert all(isinstance(name, str) for name in mask_name), "mask_name must be a list of strings"
    assert len(mask_name) == 1, "mask_name must have length 1"
    assert (
        large_mask.shape == small_mask.shape
    ), "large_mask and small_mask must have the same shape"
    subtract_mask = small_mask.image == 0
    new_mask = large_mask.image * subtract_mask
    return LabelMask(new_mask, large_mask.info, mask_name)
